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
    FixedLagSmoother     – causal average whose centre sits lag_n samples in the past

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
        Used by ``SmaSmoother`` and ``FixedLagSmoother``.
    alpha_min : float
        Minimum alpha for ``AdaptiveEmaSmoother``. Default 0.05.
    alpha_max : float
        Maximum alpha for ``AdaptiveEmaSmoother``. Default 0.30.
    variance_window_n : int
        Sample window for running variance in ``AdaptiveEmaSmoother``.
        Default 16 (= 4 s at 4 Hz).
    var_ceiling : float
        Variance level that saturates alpha at alpha_max in
        ``AdaptiveEmaSmoother``. Default 0.04 (std ≈ 0.20 on a [0,1] score).
        Calibrate from pilot data.
    lag_n : int
        Fixed lag in samples for ``FixedLagSmoother``. Default 4.
        Must be < window_n / 2. Adds lag_n * inference_step_s seconds of delay.
    """

    method: Literal["ema", "sma", "adaptive_ema", "fixed_lag"] = "ema"
    alpha: float = 0.10
    window_n: int = 8
    alpha_min: float = 0.05
    alpha_max: float = 0.30
    variance_window_n: int = 16
    var_ceiling: float = 0.04
    lag_n: int = 4


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
    """EMA whose α adapts to recent output volatility.

    When EEGNet output variance over the last *variance_window_n* samples
    is high, α is increased toward *alpha_max* (faster tracking).
    When variance is low, α decreases toward *alpha_min* (heavier smoothing).

    Mapping:
        norm_var = clip(var / var_ceiling, 0, 1)
        α = alpha_min + (alpha_max − alpha_min) · norm_var

    ⚠ Assumption: high variance = genuine rapid state change, not artefact.
      Only use if pilot data confirms this. Use EmaSmoother for first
      deployment (per ADR-0003 and implementation plan).

    Parameters
    ----------
    alpha_min : float
        Minimum α when signal is stable. Default 0.05.
    alpha_max : float
        Maximum α when signal is volatile. Default 0.30.
    variance_window_n : int
        Samples used to compute running variance. Default 16 (4 s at 4 Hz).
    var_ceiling : float
        Variance level that saturates α at alpha_max. Default 0.04.
    """

    def __init__(
        self,
        alpha_min: float = 0.05,
        alpha_max: float = 0.30,
        variance_window_n: int = 16,
        var_ceiling: float = 0.04,
    ) -> None:
        if not (0.0 < alpha_min <= alpha_max <= 1.0):
            raise ValueError(
                f"Need 0 < alpha_min ≤ alpha_max ≤ 1; "
                f"got alpha_min={alpha_min}, alpha_max={alpha_max}"
            )
        if variance_window_n < 2:
            raise ValueError(f"variance_window_n must be ≥ 2; got {variance_window_n}")
        if var_ceiling <= 0.0:
            raise ValueError(f"var_ceiling must be > 0; got {var_ceiling}")

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.variance_window_n = variance_window_n
        self.var_ceiling = var_ceiling

        self._s: Optional[float] = None
        self._var_buf: Deque[float] = deque(maxlen=variance_window_n)
        self._current_alpha: float = alpha_min

    def _running_variance(self) -> float:
        n = len(self._var_buf)
        if n < 2:
            return 0.0
        mean = sum(self._var_buf) / n
        return sum((x - mean) ** 2 for x in self._var_buf) / (n - 1)

    def update(self, value: float) -> float:
        self._var_buf.append(float(value))
        var = self._running_variance()
        norm_var = min(1.0, var / self.var_ceiling)
        self._current_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * norm_var

        if self._s is None:
            self._s = float(value)
        else:
            self._s = self._current_alpha * value + (1.0 - self._current_alpha) * self._s
        return self._s

    def reset(self) -> None:
        self._s = None
        self._var_buf.clear()
        self._current_alpha = self.alpha_min

    @property
    def current_alpha(self) -> float:
        """The α value used on the most recent update() call."""
        return self._current_alpha

    def __repr__(self) -> str:
        return (
            f"AdaptiveEmaSmoother("
            f"alpha_min={self.alpha_min}, alpha_max={self.alpha_max}, "
            f"variance_window_n={self.variance_window_n})"
        )


# ---------------------------------------------------------------------------
# FixedLagSmoother
# ---------------------------------------------------------------------------

class FixedLagSmoother(MwlSmoother):
    """Causal smoother that returns the mean of a window centred *lag_n*
    samples in the past.

    The output is delayed by lag_n * inference_step_s seconds relative to
    the most recent sample.  The average is taken over a symmetric window
    of ±lag_n samples around the lagged centre, bounded by buffer contents.

    Requires window_n > 2 * lag_n to allow a full symmetric average.

    Parameters
    ----------
    window_n : int
        Total buffer capacity (must be > 2 * lag_n). Default 8.
    lag_n : int
        Samples to look back from the newest sample as centre of average.
        Default 4. At 4 Hz: lag_n=4 → 1 s output delay.
    """

    def __init__(self, window_n: int = 8, lag_n: int = 4) -> None:
        if lag_n < 0:
            raise ValueError(f"lag_n must be ≥ 0; got {lag_n}")
        if window_n <= 2 * lag_n:
            raise ValueError(
                f"window_n ({window_n}) must be > 2 * lag_n ({2 * lag_n}) "
                "to allow a centred average at the lag position"
            )
        self.window_n = window_n
        self.lag_n = lag_n
        self._buf: Deque[float] = deque(maxlen=window_n)

    def update(self, value: float) -> float:
        self._buf.append(float(value))
        buf = list(self._buf)
        n = len(buf)

        # Centre position: lag_n samples before the newest element
        centre = n - 1 - self.lag_n
        if centre < 0:
            # Buffer not yet deep enough to reach the lag — average all available
            return sum(buf) / n

        # Symmetric window of ±lag_n around centre, clipped to buffer bounds
        lo = max(0, centre - self.lag_n)
        hi = min(n, centre + self.lag_n + 1)
        window_slice = buf[lo:hi]
        return sum(window_slice) / len(window_slice)

    def reset(self) -> None:
        self._buf.clear()

    def __repr__(self) -> str:
        return f"FixedLagSmoother(window_n={self.window_n}, lag_n={self.lag_n})"


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
            variance_window_n=cfg.variance_window_n,
            var_ceiling=cfg.var_ceiling,
        )
    elif cfg.method == "fixed_lag":
        return FixedLagSmoother(window_n=cfg.window_n, lag_n=cfg.lag_n)
    else:
        raise ValueError(
            f"Unknown smoother method {cfg.method!r}. "
            "Expected one of: 'ema', 'sma', 'adaptive_ema', 'fixed_lag'."
        )
