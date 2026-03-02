"""staircase_controller.py

1-up / 1-down staircase difficulty controller with:
  - sliding-window performance evaluation
  - cooldown enforcement between steps
  - reversal-based step-size halving

Terminology
-----------
*score* is a normalised performance metric in [0, 1].
The caller decides what it means for each task (e.g. proportion of comms
prompts answered correctly, or 1 − normalised tracking RMSE).

Direction
---------
Difficulty steps UP (harder) when window mean score > target + tolerance.
Difficulty steps DOWN (easier) when window mean score < target − tolerance.
The returned delta is applied by the caller to DifficultyState.

Usage
-----
    ctrl = StaircaseController(target_score=0.70, window_sec=45.0)
    ctrl.push_performance(t=12.3, score=0.85)
    ...
    delta = ctrl.tick(t=50.0)   # returns +0.05, −0.05, or None
    if delta is not None:
        state.update(state.d + delta)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class _Sample:
    t: float
    score: float


class StaircaseController:
    """Sliding-window 1-up/1-down staircase controller.

    Parameters
    ----------
    target_score:
        Desired normalised performance level in [0, 1].  Default 0.70.
    tolerance:
        Half-width of the dead band around *target_score*; no step is taken
        when |window_mean − target| ≤ tolerance.  Default 0.05.
    window_sec:
        Duration of the rolling evaluation window in seconds.  Default 45.
    min_samples:
        Minimum number of samples that must be present in the window before
        any step is allowed.  Default 3.
    step_up:
        Difficulty increment when stepping up (score too high → harder).
        Expressed as a fraction of the [0, 1] difficulty range.  Default 0.05.
    step_down:
        Difficulty decrement when stepping down (score too low → easier).
        Default 0.05.
    cooldown_sec:
        Minimum scenario time between successive steps.  Default 20 s.
    n_reversals_to_halve:
        After this many direction reversals step_up and step_down are each
        halved (never below *min_step*).  Set to 0 to disable.  Default 4.
    min_step:
        Floor applied when halving step sizes.  Default 0.01.
    """

    def __init__(
        self,
        target_score: float = 0.70,
        tolerance: float = 0.05,
        window_sec: float = 45.0,
        min_samples: int = 3,
        step_up: float = 0.05,
        step_down: float = 0.05,
        cooldown_sec: float = 20.0,
        n_reversals_to_halve: int = 4,
        min_step: float = 0.01,
    ) -> None:
        if not (0.0 <= target_score <= 1.0):
            raise ValueError(f"target_score must be in [0, 1]; got {target_score}")
        if step_up <= 0 or step_down <= 0:
            raise ValueError("step_up and step_down must be positive")
        if min_step <= 0:
            raise ValueError("min_step must be positive")

        self.target_score = target_score
        self.tolerance = tolerance
        self.window_sec = window_sec
        self.min_samples = min_samples
        self.step_up = step_up
        self.step_down = step_down
        self.cooldown_sec = cooldown_sec
        self.n_reversals_to_halve = n_reversals_to_halve
        self.min_step = min_step

        self._buffer: Deque[_Sample] = deque()
        self._last_step_t: float = -999.0
        self._last_direction: Optional[int] = None  # +1 or -1
        self._reversal_count: int = 0
        self._first_sample_t: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push_performance(self, t: float, score: float) -> None:
        """Add a performance sample at scenario time *t*.

        Scores should be in [0, 1] but are not hard-clipped here so that
        callers can detect out-of-range values in their own validation.
        """
        if self._first_sample_t is None:
            self._first_sample_t = float(t)
        self._buffer.append(_Sample(t=float(t), score=float(score)))

    def tick(self, t: float) -> Optional[float]:
        """Evaluate the current window and return a difficulty delta or None.

        Call once per frame (or per adaptation check interval).

        Returns
        -------
        +step_up   if window mean is above target + tolerance  (make harder)
        −step_down if window mean is below target − tolerance  (make easier)
        None       otherwise (no change warranted)
        """
        # Drop samples older than the window
        cutoff = t - self.window_sec
        while self._buffer and self._buffer[0].t < cutoff:
            self._buffer.popleft()

        # Need enough samples
        if len(self._buffer) < self.min_samples:
            return None

        # Require at least window_sec of session time before allowing first step
        # (avoids jumping on sparse early data)
        if self._first_sample_t is None:
            return None
        if (t - self._first_sample_t) < self.window_sec:
            return None

        # Enforce cooldown
        if (t - self._last_step_t) < self.cooldown_sec:
            return None

        mean_score = sum(s.score for s in self._buffer) / len(self._buffer)

        if mean_score > self.target_score + self.tolerance:
            direction = +1
        elif mean_score < self.target_score - self.tolerance:
            direction = -1
        else:
            return None

        # Reversal detection and step-size halving
        if self._last_direction is not None and direction != self._last_direction:
            self._reversal_count += 1
            if (
                self.n_reversals_to_halve > 0
                and self._reversal_count % self.n_reversals_to_halve == 0
            ):
                self.step_up = max(self.min_step, self.step_up / 2.0)
                self.step_down = max(self.min_step, self.step_down / 2.0)

        self._last_direction = direction
        self._last_step_t = t

        return self.step_up if direction == +1 else -self.step_down

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def reversal_count(self) -> int:
        """Total number of direction reversals since construction."""
        return self._reversal_count

    @property
    def window_mean(self) -> Optional[float]:
        """Mean score over current window contents; None if buffer empty."""
        if not self._buffer:
            return None
        return sum(s.score for s in self._buffer) / len(self._buffer)

    @property
    def n_samples_in_window(self) -> int:
        return len(self._buffer)

    def as_dict(self) -> dict:
        """Serialisable snapshot for logging."""
        return {
            "target_score": self.target_score,
            "tolerance": self.tolerance,
            "window_sec": self.window_sec,
            "step_up": round(self.step_up, 6),
            "step_down": round(self.step_down, 6),
            "cooldown_sec": self.cooldown_sec,
            "reversal_count": self._reversal_count,
            "n_samples": len(self._buffer),
            "window_mean": (
                round(self.window_mean, 4) if self.window_mean is not None else None
            ),
        }

    def __repr__(self) -> str:
        return (
            f"StaircaseController("
            f"target={self.target_score}, "
            f"window={self.window_sec}s, "
            f"step_up={self.step_up:.3f}, step_down={self.step_down:.3f}, "
            f"reversals={self._reversal_count})"
        )
