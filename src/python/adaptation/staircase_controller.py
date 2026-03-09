"""staircase_controller.py

1-up / 1-down staircase difficulty controller with:
  - sliding-window performance evaluation
  - cooldown enforcement between steps
  - graduated step-size schedule (coarse → fine on each direction reversal)
  - convergence detection: N consecutive no-step ticks at the finest step

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

Step schedule
-------------
The staircase starts with the largest step in *step_schedule*.  Each time
the step direction reverses the controller advances to the next (smaller)
entry.  Once the final entry is reached, it remains fixed.

Convergence
-----------
After the step size has reached the final (smallest) schedule entry,
*stable_ticks_required* consecutive ``tick()`` calls that fire no step
(score within the dead band) set ``converged = True``.  Subsequent calls
to ``tick()`` always return ``None`` — the caller should freeze difficulty
and end the block.

Usage
-----
    ctrl = StaircaseController(target_score=0.70, window_sec=45.0)
    ctrl.push_performance(t=12.3, score=0.85)
    ...
    delta = ctrl.tick(t=50.0)   # returns +step, −step, or None
    if ctrl.converged:
        # difficulty has stabilised — end session
        ...
    elif delta is not None:
        state.update(state.d + delta)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


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
    step_schedule:
        Ordered tuple of step sizes (fraction of [0, 1] difficulty range).
        The first entry is used initially; each direction reversal advances
        to the next entry.  The last entry is the final fine step and is
        never made smaller.  Default (0.2, 0.1, 0.05).
    cooldown_sec:
        Minimum scenario time between successive steps.  Default 20 s.
    stable_ticks_required:
        Number of consecutive no-step tick() calls (at the finest step)
        required to declare convergence.  Default 3.

    Legacy parameters (accepted but silently ignored)
    --------------------------------------------------
    step_up, step_down, n_reversals_to_halve, min_step
        These were present in an earlier API.  Pass them freely; they have
        no effect on behaviour.
    """

    def __init__(
        self,
        target_score: float = 0.70,
        tolerance: float = 0.05,
        window_sec: float = 45.0,
        min_samples: int = 3,
        step_schedule: Tuple[float, ...] = (0.2, 0.1, 0.05),
        cooldown_sec: float = 20.0,
        stable_ticks_required: int = 3,
        # Legacy kwargs kept so old call sites don't raise TypeError.
        step_up: float = 0.0,
        step_down: float = 0.0,
        n_reversals_to_halve: int = 0,
        min_step: float = 0.0,
    ) -> None:
        if not (0.0 <= target_score <= 1.0):
            raise ValueError(f"target_score must be in [0, 1]; got {target_score}")
        if not step_schedule:
            raise ValueError("step_schedule must be a non-empty sequence of positive floats")
        for s in step_schedule:
            if s <= 0:
                raise ValueError(f"All step_schedule entries must be positive; got {s}")

        self.target_score = target_score
        self.tolerance = tolerance
        self.window_sec = window_sec
        self.min_samples = min_samples
        self.cooldown_sec = cooldown_sec
        self.stable_ticks_required = stable_ticks_required

        self._step_schedule: Tuple[float, ...] = tuple(step_schedule)
        self._schedule_idx: int = 0                  # current position in schedule
        self._stable_ticks_at_final_step: int = 0    # consecutive no-step ticks at finest step
        self._converged: bool = False

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

        Call once per adaptation check interval.

        Returns
        -------
        +step  if window mean is above target + tolerance  (make harder)
        −step  if window mean is below target − tolerance  (make easier)
        None   otherwise (no change warranted, or already converged)

        Side effects
        ------------
        Sets ``self.converged = True`` once *stable_ticks_required*
        consecutive no-step ticks occur at the finest step size.
        """
        if self._converged:
            return None

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

        at_final_step = (self._schedule_idx >= len(self._step_schedule) - 1)

        if mean_score > self.target_score + self.tolerance:
            direction = +1
        elif mean_score < self.target_score - self.tolerance:
            direction = -1
        else:
            # Score is within the dead band — no step.
            # Count stability only once we are on the final fine step.
            if at_final_step:
                self._stable_ticks_at_final_step += 1
                if self._stable_ticks_at_final_step >= self.stable_ticks_required:
                    self._converged = True
            return None

        # A step will fire.
        # If we're already at the final step, reset the stability counter.
        if at_final_step:
            self._stable_ticks_at_final_step = 0

        # Reversal detection: advance the step schedule on each direction flip.
        if self._last_direction is not None and direction != self._last_direction:
            self._reversal_count += 1
            if self._schedule_idx < len(self._step_schedule) - 1:
                self._schedule_idx += 1

        self._last_direction = direction
        self._last_step_t = t

        current_step = self._step_schedule[self._schedule_idx]
        return current_step if direction == +1 else -current_step

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_up(self) -> float:
        """Current step size for upward moves (mirrors current schedule entry)."""
        return self._step_schedule[self._schedule_idx]

    @property
    def step_down(self) -> float:
        """Current step size for downward moves (mirrors current schedule entry)."""
        return self._step_schedule[self._schedule_idx]

    @property
    def converged(self) -> bool:
        """True once the staircase has stabilised at the finest step."""
        return self._converged

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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """Serialisable snapshot for logging."""
        return {
            "target_score": self.target_score,
            "tolerance": self.tolerance,
            "window_sec": self.window_sec,
            "step_schedule": list(self._step_schedule),
            "step_schedule_idx": self._schedule_idx,
            "step_up": round(self.step_up, 6),
            "step_down": round(self.step_down, 6),
            "cooldown_sec": self.cooldown_sec,
            "reversal_count": self._reversal_count,
            "stable_ticks_required": self.stable_ticks_required,
            "stable_ticks_at_final_step": self._stable_ticks_at_final_step,
            "converged": self._converged,
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
            f"step={self.step_up:.3f} (schedule idx {self._schedule_idx}), "
            f"reversals={self._reversal_count}, "
            f"converged={self._converged})"
        )
