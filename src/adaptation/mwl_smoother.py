"""mwl_smoother.py

Online smoothers for EEGNet overload_score output.

All smoothers implement the same two-method interface so they are
interchangeable in the EEG adaptation pipeline:

    smoother.update(value: float) -> float   # consume one sample, return smoothed value
    smoother.reset() -> None                 # reset to uninitialised state

Four implementations are provided:

    EmaSmoother          – exponential moving average (recommended default)
    SmaSmoother          – simple moving average over a fixed sample window
    AdaptiveEmaSmoother  – EMA whose α grows when recent variance is high
    FixedLagSmoother     – Kalman fixed-lag smoother with RTS backward pass

Recommended default: EmaSmoother(alpha=0.10)
    At 4 Hz inference rate this gives an effective time constant of ~22 samples
    (~5.5 s), which is appropriate for the 5 s adaptation check interval.
    This is the value specified in ADR-0003.

No vendor / LSL / torch imports — this module is safe to import and test
outside the OpenMATB bootstrap subprocess.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class MwlSmootherConfig:
    """Configuration for all smoother types.

    Parameters
    ----------
    method : str
        Which smoother to construct. One of:
        ``"ema"`` | ``"sma"`` | ``"adaptive_ema"`` | ``"fixed_lag"``
    alpha : float
        EMA decay coefficient ∈ (0, 1]. Default 0.10.
        Used by ``EmaSmoother`` and as the starting value for
        ``AdaptiveEmaSmoother``.
        Effective time constant ≈ (1/alpha − 1) samples.
        At 4 Hz inference: alpha=0.10 → τ ≈ 22 samples ≈ 5.5 s.
    window_n : int
        Number of samples in the rolling window. Default 8 (= 2 s at 4 Hz).
        Used by ``SmaSmoother``.
    alpha_min : float
        Minimum alpha for ``AdaptiveEmaSmoother``. Default 0.05.
        Used when classifier confidence is lowest (p ≈ 0.5).
    alpha_max : float
        Maximum alpha for ``AdaptiveEmaSmoother``. Default 0.30.
        Used when classifier confidence is highest (p ≈ 0 or 1).
    lag_n : int
        Fixed lag in samples for ``FixedLagSmoother``. Default 4.
        Adds lag_n * inference_step_s seconds of output delay.
    process_noise : float
        Process noise variance (Q) for ``FixedLagSmoother``. Default 0.005.
        Controls expected step-to-step change in the latent workload state.
    measurement_noise : float
        Measurement noise variance (R) for ``FixedLagSmoother``. Default 0.1.
        Controls how much individual classifier outputs are trusted.
    """

    method: Literal["ema", "sma", "adaptive_ema", "fixed_lag"] = "ema"
    alpha: float = 0.10
    window_n: int = 8
    alpha_min: float = 0.05
    alpha_max: float = 0.30
    lag_n: int = 4
    process_noise: float = 0.005
    measurement_noise: float = 0.1


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class MwlSmoother:
    """Abstract base class defining the smoother interface."""

    def update(self, value: float) -> float:
        """Consume one raw sample and return the current smoothed value."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state. The next call to update() starts fresh."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# EmaSmoother
# ---------------------------------------------------------------------------

class EmaSmoother(MwlSmoother):
    """Exponential moving average.

    s_t = α · x_t + (1 − α) · s_{t-1}

    The first sample initialises the internal state directly (no ramp-in bias).

    Parameters
    ----------
    alpha : float
        Decay coefficient ∈ (0, 1]. Larger = reacts faster, less smoothing.
        ADR-0003 default: 0.10.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self._s: Optional[float] = None

    def update(self, value: float) -> float:
        if self._s is None:
            self._s = float(value)
        else:
            self._s = self.alpha * value + (1.0 - self.alpha) * self._s
        return self._s

    def reset(self) -> None:
        self._s = None

    def __repr__(self) -> str:
        return f"EmaSmoother(alpha={self.alpha})"


# ---------------------------------------------------------------------------
# SmaSmoother
# ---------------------------------------------------------------------------

class SmaSmoother(MwlSmoother):
    """Simple moving average over the last *window_n* samples.

    Before the buffer fills, returns the mean of however many samples
    have been seen. Once full, a true equal-weight rolling average.

    Parameters
    ----------
    window_n : int
        Number of samples in the rolling window. Default 8.
    """

    def __init__(self, window_n: int = 8) -> None:
        if window_n < 1:
            raise ValueError(f"window_n must be ≥ 1; got {window_n}")
        self.window_n = window_n
        self._buf: Deque[float] = deque(maxlen=window_n)

    def update(self, value: float) -> float:
        self._buf.append(float(value))
        return sum(self._buf) / len(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def __repr__(self) -> str:
        return f"SmaSmoother(window_n={self.window_n})"


# ---------------------------------------------------------------------------
# AdaptiveEmaSmoother
# ---------------------------------------------------------------------------

class AdaptiveEmaSmoother(MwlSmoother):
    """Confidence-weighted adaptive causal EMA filter.

    Applies recursive exponential smoothing where the smoothing coefficient
    α is dynamically adjusted as a function of classifier confidence.
    Classifier confidence is derived from the input probability:

        confidence = 2 · |p − 0.5|      ∈ [0, 1]

    Mapping to α:
        α = alpha_min + (alpha_max − alpha_min) · confidence

    When the classifier is confident (p near 0 or 1), α is high and the
    filter tracks new observations closely.  When the classifier is
    uncertain (p near 0.5), α is low and the filter relies more on its
    history, stabilising transient fluctuations.

    Operates strictly causally with zero imposed latency.

    Parameters
    ----------
    alpha_min : float
        Minimum α when classifier confidence is lowest (p ≈ 0.5). Default 0.05.
    alpha_max : float
        Maximum α when classifier confidence is highest (p ≈ 0 or 1). Default 0.30.
    """

    def __init__(
        self,
        alpha_min: float = 0.05,
        alpha_max: float = 0.30,
    ) -> None:
        if not (0.0 < alpha_min <= alpha_max <= 1.0):
            raise ValueError(
                f"Need 0 < alpha_min ≤ alpha_max ≤ 1; "
                f"got alpha_min={alpha_min}, alpha_max={alpha_max}"
            )

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self._s: Optional[float] = None
        self._current_alpha: float = alpha_min

    def update(self, value: float) -> float:
        confidence = 2.0 * abs(float(value) - 0.5)
        self._current_alpha = (
            self.alpha_min + (self.alpha_max - self.alpha_min) * confidence
        )

        if self._s is None:
            self._s = float(value)
        else:
            self._s = self._current_alpha * value + (1.0 - self._current_alpha) * self._s
        return self._s

    def reset(self) -> None:
        self._s = None
        self._current_alpha = self.alpha_min

    @property
    def current_alpha(self) -> float:
        """The α value used on the most recent update() call."""
        return self._current_alpha

    def __repr__(self) -> str:
        return (
            f"AdaptiveEmaSmoother("
            f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max})"
        )


# ---------------------------------------------------------------------------
# FixedLagSmoother
# ---------------------------------------------------------------------------

class FixedLagSmoother(MwlSmoother):
    """Kalman fixed-lag smoother for scalar workload probability.

    Models the latent workload state as a random walk observed through
    noisy classifier outputs:

        x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   (process model)
        z_t = x_t + v_t,        v_t ~ N(0, R)   (measurement model)

    At each time step the Kalman filter runs a forward prediction/update
    pass.  A buffer of the last ``lag_n + 1`` filtered states is maintained.
    Rauch–Tung–Striebel (RTS) backward smoothing is then applied over the
    buffer to return the smoothed estimate at time ``t − lag_n``.

    This introduces an explicit, bounded delay of ``lag_n`` samples while
    reducing posterior variance compared to purely causal filtering.

    Parameters
    ----------
    lag_n : int
        Fixed lag in samples. The output estimates x_{t − lag_n} given
        observations up to t. Default 4 (1 s at 4 Hz).
    process_noise : float
        Process noise variance Q.  Controls expected step-to-step change
        in the latent workload state. Default 0.005.
    measurement_noise : float
        Measurement noise variance R.  Controls how much individual
        classifier outputs are trusted. Default 0.1.
    """

    def __init__(
        self,
        lag_n: int = 4,
        process_noise: float = 0.005,
        measurement_noise: float = 0.1,
    ) -> None:
        if lag_n < 0:
            raise ValueError(f"lag_n must be >= 0; got {lag_n}")
        if process_noise <= 0:
            raise ValueError(f"process_noise must be > 0; got {process_noise}")
        if measurement_noise <= 0:
            raise ValueError(f"measurement_noise must be > 0; got {measurement_noise}")
        self.lag_n = lag_n
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        # Forward-filtered state buffers (ring buffer, size lag_n + 1)
        self._x_fwd: Deque[float] = deque(maxlen=max(lag_n + 1, 1))
        self._P_fwd: Deque[float] = deque(maxlen=max(lag_n + 1, 1))
        # Current Kalman filter state
        self._x: Optional[float] = None
        self._P: Optional[float] = None

    def update(self, value: float) -> float:
        z = float(value)
        Q = self.process_noise
        R = self.measurement_noise

        if self._x is None:
            # Initialise filter with first observation
            self._x = z
            self._P = R
        else:
            # Kalman predict (random walk: predicted state = previous state)
            P_pred = self._P + Q
            # Kalman update
            K = P_pred / (P_pred + R)
            self._x = self._x + K * (z - self._x)
            self._P = (1.0 - K) * P_pred

        # Store forward-filtered state
        self._x_fwd.append(self._x)
        self._P_fwd.append(self._P)

        if self.lag_n == 0:
            return max(0.0, min(1.0, self._x))

        # Not enough history yet — return oldest available filtered estimate
        if len(self._x_fwd) <= self.lag_n:
            return max(0.0, min(1.0, self._x_fwd[0]))

        # RTS backward smoothing: start from newest, smooth back to index 0
        xs = list(self._x_fwd)
        Ps = list(self._P_fwd)
        n = len(xs)

        x_s = xs[n - 1]
        P_s = Ps[n - 1]
        for k in range(n - 2, -1, -1):
            P_pred_k1 = Ps[k] + Q
            G = Ps[k] / P_pred_k1          # smoother gain
            x_s = xs[k] + G * (x_s - xs[k])
            P_s = Ps[k] + G * G * (P_s - P_pred_k1)

        return max(0.0, min(1.0, x_s))

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._x_fwd.clear()
        self._P_fwd.clear()

    def __repr__(self) -> str:
        return (
            f"FixedLagSmoother(lag_n={self.lag_n}, "
            f"Q={self.process_noise}, R={self.measurement_noise})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_smoother(cfg: MwlSmootherConfig) -> MwlSmoother:
    """Construct a smoother from *cfg*.

    Uses ``cfg.method`` to select the implementation and the relevant
    fields for that type.

    Returns
    -------
    MwlSmoother
        Freshly constructed, uninitialised smoother instance.

    Raises
    ------
    ValueError
        If ``cfg.method`` is not one of the four supported values.
    """
    if cfg.method == "ema":
        return EmaSmoother(alpha=cfg.alpha)
    elif cfg.method == "sma":
        return SmaSmoother(window_n=cfg.window_n)
    elif cfg.method == "adaptive_ema":
        return AdaptiveEmaSmoother(
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
        )
    elif cfg.method == "fixed_lag":
        return FixedLagSmoother(
            lag_n=cfg.lag_n,
            process_noise=cfg.process_noise,
            measurement_noise=cfg.measurement_noise,
        )
    else:
        raise ValueError(
            f"Unknown smoother method {cfg.method!r}. "
            "Expected one of: 'ema', 'sma', 'adaptive_ema', 'fixed_lag'."
        )
