"""difficulty_state.py

Single source of truth for the continuous difficulty scalar d ∈ [0, 1] and
all derived task parameters used during online staircase calibration.

All parameter mappings are direct ports of generate_pilot_scenarios.py so that
online-adaptation sessions are quantitatively comparable with the pre-generated
pilot scenarios.

Usage
-----
    state = DifficultyState(d_init=0.5, seed=42)
    state.update(0.65)
    print(state.params.track_joystick_force)   # → 1.7
    print(state.as_dict())                     # → serialisable snapshot for logging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constant parameter bounds
# ---------------------------------------------------------------------------
# Tracking – matched to add_scenario_phase() in generate_pilot_scenarios.py
#
# Hard endpoints are set to the theoretical performance limits of the task:
#
#   _TRACK_FORCE_HARD = 1.0
#     The joystick physics applies `x_input * joystickforce` pixels per step as
#     correction.  Pyglet normalises physical axes to ±1.0, so max correction
#     per step = 1.0 * joystickforce.  Peak sinusoidal y-drift per step is
#     xgain * 0.006 ≈ 160 * 0.006 = 0.96 px/step (for a ~400 px wide reticle).
#     At joystickforce=1.0, a perfect user can just barely cancel peak drift.
#     Values below 1.0 make perfect compensation physically impossible and
#     were removed from the previous 0.59 hard endpoint (a 30%-more-extreme
#     extrapolation beyond the calibrated pilot HIGH).
#
#   _TRACK_UPDATE_HARD_MS = 10.0
#     Reverts to the original OpenMATB design value.  The previous 5.9 ms was
#     an untested 30% extrapolation beyond the calibrated pilot HIGH.
#     At 10 ms the cursor updates 100 ×/s; both drift and correction scale
#     equally with step rate so this does not affect the difficulty balance.
_TRACK_UPDATE_EASY_MS: float = 50.0   # d=-0.8: slow cursor update rate (easiest, floor participant LOW)
_TRACK_UPDATE_HARD_MS: float = 10.0   # d=+1.8: fast cursor update rate (hardest, ceiling participant HIGH)
_TRACK_FORCE_EASY: float = 3.0        # d=-0.8: strong joystick correction (easiest)
_TRACK_FORCE_HARD: float = 1.0        # d=+1.8: compensation limit — a perfect user can just cancel peak drift

# ---------------------------------------------------------------------------
# Log-scale event-rate and drain constants
# ---------------------------------------------------------------------------
# All event rates and drain use a common log (exponential) scale so that the
# HIGH/LOW fold-change is the same for every participant regardless of where
# the staircase converges.
#
# Design endpoints:
#   D_LOG_MIN = -0.8  →  floor participant LOW level  (d_final=0, delta=0.8)
#   D_LOG_MAX = +1.8  →  ceiling participant HIGH level (d_final=1, delta=0.8)
#   Total range = 2.6
#
# SysMon (lights + scales) — no physical scheduling cap:
#   1 event/block at d=-0.8,  18 events/block at d=+1.8
#   Gives exact 6x H/L for every participant (delta=0.8, range=2.6):
#     fold = (18/1)^(2*0.8/2.6) = 18^0.615 ≈ 6
#
# Comms — physically capped by prompt duration (18s) + refractory (1s) = 19s/slot:
#   floor(54/19) = 2 prompts/block maximum.
#   1 event/block at d=-0.8,  2 events/block at d=+1.8.
#   Limited separation (1→2 per block) but 1 vs 2 concurrent prompts still
#   loads the participant differently due to task interaction.
#
# ResMan pump failures — capped by failure duration (10s) + refractory (1s) = 11s/slot:
#   floor(54/11) = 4 failures/block maximum.
#   1 event/block at d=-0.8,  4 events/block at d=+1.8.
#
# Drain (continuous leak) — log scale, 50 ml/min at d=-0.8, 1200 ml/min at d=+1.8.
#   1200 ml/min is the physical pump-network maximum.

_D_LOG_MIN: float = -0.8   # d value at the low anchor (floor participant LOW level)
_D_LOG_MAX: float =  1.8   # d value at the high anchor (ceiling participant HIGH level)
_D_LOG_RANGE: float = _D_LOG_MAX - _D_LOG_MIN   # 2.6

# Event rates (Hz): value = MIN * (MAX/MIN)^t  where t = (d - D_MIN) / D_RANGE
_EFF_SEC: float = 54.0                          # schedulable seconds per 60-s block
_SYSMON_RATE_MIN_HZ:  float =  1.0 / _EFF_SEC  # 1 event/block at d=-0.8
_SYSMON_RATE_MAX_HZ:  float = 18.0 / _EFF_SEC  # 18 events/block at d=+1.8  (6x H/L)
_COMMS_RATE_MIN_HZ:   float =  1.0 / _EFF_SEC  # 1 event/block at d=-0.8
_COMMS_RATE_MAX_HZ:   float =  2.0 / _EFF_SEC  # 2 events/block at d=+1.8  (physical max)
_PUMP_RATE_MIN_HZ:    float =  1.0 / _EFF_SEC  # 1 event/block at d=-0.8
_PUMP_RATE_MAX_HZ:    float =  4.0 / _EFF_SEC  # 4 events/block at d=+1.8  (physical max)

# Drain (ml/min)
# The naive-participant sustainable max inflow to tank A or B is 600 ml/min
# (pump 2: E→A and pump 4: F→B, drawing from infinite tanks E and F).
# Pumps 1 and 3 draw from finite reserves C and D which deplete in ~1.25 min
# without also running pumps 5/6; participants cannot be assumed to know this.
# Ceiling is set to 400 ml/min — safely below the 600 ml/min naive max —
# so tanks are always recoverable.  The pump failure events (up to 4/block at
# d=+1.8) provide the primary resman workload challenge at high difficulty.
_DRAIN_MIN_ML_MIN: float =  50.0   # at d=-0.8
_DRAIN_MAX_ML_MIN: float = 400.0   # at d=+1.8 (below naive sustainable max of 600 ml/min)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lerp(easy: float, hard: float, d: float) -> float:
    """Linear interpolation between *easy* (d=0) and *hard* (d=1)."""
    return easy + (hard - easy) * d


def _log_rate(min_val: float, max_val: float, d: float) -> float:
    """Exponential interpolation anchored at _D_LOG_MIN (-0.8) and _D_LOG_MAX (+1.8).

    t = 0 at d=-0.8, t = 1 at d=+1.8.  Clamped so values outside that range
    return the endpoint value (no further extrapolation).
    """
    t = max(0.0, min(1.0, (d - _D_LOG_MIN) / _D_LOG_RANGE))
    return min_val * (max_val / min_val) ** t


def _log_drain(d: float) -> int:
    """Continuous resman tank drain (ml/min).  Log scale, 50->1200 over [-0.8, +1.8]."""
    return round(_log_rate(_DRAIN_MIN_ML_MIN, _DRAIN_MAX_ML_MIN, d))


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskParams:
    """All task parameters derived from a single difficulty scalar d."""

    # Tracking
    track_update_ms: float
    track_joystick_force: float

    # ResMan continuous drain (ml/min per target tank)
    resman_loss_a_per_min: int
    resman_loss_b_per_min: int

    # Event-generator rates (Hz) — consumed by PoissonEventGenerator
    comms_rate_hz: float
    sysmon_light_rate_hz: float
    sysmon_scale_rate_hz: float
    resman_pump_rate_hz: float


def make_task_params(d: float) -> TaskParams:
    """Compute TaskParams for difficulty d.  Valid for any real d.

    Track and joystick remain linear (d=0 easy, d=1 hard) with physical floors.
    All event rates and drain use a log (exponential) scale anchored at
    d=-0.8 (minimum) and d=+1.8 (maximum), giving ~6x H/L for sysmon and
    consistent fold-change regardless of where the staircase converges.
    Log-scale values are clamped at their endpoints for d outside [-0.8, 1.8].
    """
    # Tracking uses the same [-0.8, +1.8] range as all other subtasks.  t=0 at
    # d=-0.8 (easiest: 50 ms update, force 3.0) and t=1 at d=+1.8 (hardest:
    # 10 ms update, force 1.0).  Clamped at endpoints — no extrapolation.
    t_track = max(0.0, min(1.0, (d - _D_LOG_MIN) / _D_LOG_RANGE))
    return TaskParams(
        track_update_ms=_lerp(_TRACK_UPDATE_EASY_MS, _TRACK_UPDATE_HARD_MS, t_track),
        track_joystick_force=_lerp(_TRACK_FORCE_EASY, _TRACK_FORCE_HARD, t_track),
        resman_loss_a_per_min=_log_drain(d),
        resman_loss_b_per_min=_log_drain(d),
        comms_rate_hz=_log_rate(_COMMS_RATE_MIN_HZ, _COMMS_RATE_MAX_HZ, d),
        sysmon_light_rate_hz=_log_rate(_SYSMON_RATE_MIN_HZ, _SYSMON_RATE_MAX_HZ, d),
        sysmon_scale_rate_hz=_log_rate(_SYSMON_RATE_MIN_HZ, _SYSMON_RATE_MAX_HZ, d),
        resman_pump_rate_hz=_log_rate(_PUMP_RATE_MIN_HZ, _PUMP_RATE_MAX_HZ, d),
    )


# ---------------------------------------------------------------------------
# DifficultyState
# ---------------------------------------------------------------------------

class DifficultyState:
    """Holds the current difficulty scalar and all derived task parameters.

    Parameters
    ----------
    d_init:
        Starting difficulty in [0, 1].  Default 0.5.
    d_min, d_max:
        Hard clamps applied on every call to ``update()``.
    seed:
        Stored for reproducibility logging and passed to event generators;
        not consumed internally by this class.
    """

    def __init__(
        self,
        d_init: float = 0.5,
        d_min: float = 0.0,
        d_max: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        # d_min / d_max can be outside [0, 1] when the staircase needs to find
        # a threshold for very easy (d_min < 0) or ceiling (d_max > 1)
        # participants.  make_task_params() extrapolates linearly and applies
        # physical floors/ceilings to keep all output values valid.
        if d_min > d_max:
            raise ValueError(
                f"Invalid difficulty bounds: d_min={d_min} > d_max={d_max}"
            )

        self.d_min = d_min
        self.d_max = d_max
        self.seed = seed

        # Initialised in update()
        self.d: float = 0.0
        self.d_prev: float = 0.0
        self.params: TaskParams = make_task_params(0.5)

        self.update(d_init)

    # ------------------------------------------------------------------

    def update(self, d_new: float) -> None:
        """Set a new difficulty level, clamped to [d_min, d_max].

        Stores the previous value in ``d_prev`` for logging delta.
        """
        self.d_prev = self.d
        self.d = max(self.d_min, min(self.d_max, float(d_new)))
        self.params = make_task_params(self.d)

    def as_dict(self) -> dict:
        """Return a fully serialisable snapshot for structured logging."""
        return {
            "d": round(self.d, 6),
            "d_prev": round(self.d_prev, 6),
            "track_update_ms": self.params.track_update_ms,
            "track_joystick_force": round(self.params.track_joystick_force, 4),
            "resman_loss_a_per_min": self.params.resman_loss_a_per_min,
            "resman_loss_b_per_min": self.params.resman_loss_b_per_min,
            "comms_rate_hz": round(self.params.comms_rate_hz, 6),
            "sysmon_light_rate_hz": round(self.params.sysmon_light_rate_hz, 6),
            "sysmon_scale_rate_hz": round(self.params.sysmon_scale_rate_hz, 6),
            "resman_pump_rate_hz": round(self.params.resman_pump_rate_hz, 6),
        }

    def __repr__(self) -> str:
        return (
            f"DifficultyState(d={self.d:.3f}, "
            f"d_min={self.d_min}, d_max={self.d_max})"
        )
