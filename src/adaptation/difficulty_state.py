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
_TRACK_UPDATE_EASY_MS: float = 50.0   # d=0: slow cursor update rate (easier)
_TRACK_UPDATE_HARD_MS: float = 10.0   # d=1: fast cursor update rate (original design value)
_TRACK_FORCE_EASY: float = 3.0        # d=0: strong joystick correction (easier)
_TRACK_FORCE_HARD: float = 1.0        # d=1: compensation limit — a perfect user can just cancel peak drift

# ResMan continuous drain – pump network totals from vendor resman.py defaults:
#   infinite-capacity pumps 2+4 : 600+600  = 1200 ml/min
#   finite-capacity pumps  5+6  : 600+600  = 1200 ml/min
#   maximum_single_leakage      = (1200+1200)/2 = 1200 ml/min
# _RESMAN_MAX_LEAK_PER_MIN matches generate_pilot_scenarios.py: maximum_single_leakage=1200.
# Key values (with two-step offset): d=0.20→240, d=0.55→560, d=0.95→940 ml/min.
_RESMAN_MAX_LEAK_PER_MIN: int = 1200
_RESMAN_OFFSET_D_THRESHOLD: float = 0.50   # subtract 100 ml/min above this d
_RESMAN_OFFSET_MODERATE: int = -100
_RESMAN_OFFSET_HIGH_THRESHOLD: float = 0.90  # subtract a further 100 above this d
_RESMAN_OFFSET_HIGH: int = -100

# Comms event rate – TOTAL rate across all comms channels.
#   build_standard_generators splits it equally: each of the 2 channels (own/other)
#   gets rate * 0.5.  So total events per 300s = rate * 300.
#   Calibrated against pilot scenario files:
#     LOW  (d=0.20):  3 prompts  → rate = 3/300 = 0.010  Hz
#     MOD  (d=0.55):  8 prompts  → rate = 8/300 = 0.027  Hz  (lerp gives ~0.030)
#     HIGH (d=0.95): 14 prompts  → rate = 14/300 ≈ 0.047 Hz  (lerp gives ~0.048)
#   Hard endpoint stretched 30% beyond the original d=1 calibration value of 0.050:
#     d=1 → 0.050 * 1.3 = 0.065 → ~19.5 prompts/300s
_COMMS_RATE_EASY_HZ: float = 0.005   # ~1.5 prompts/300 s at d=0
_COMMS_RATE_HARD_HZ: float = 0.065   # 0.050 * 1.3; ~19.5 total prompts/300 s at d=1

# SysMon event rate – TOTAL rate used identically for lights and scales groups.
#   build_standard_generators splits each group: lights ÷ 2, scales ÷ 4.
#   Total events per 300s = (sysmon_light_rate + sysmon_scale_rate) * 300
#                         = 2 * sysmon_rate * 300.
#   Calibrated against pilot scenario files:
#     LOW  (d=0.20): 10 failures → rate = 10/600 = 0.017 Hz  (lerp gives ~0.024)
#     MOD  (d=0.55): 30 failures → rate = 30/600 = 0.050 Hz  (lerp gives ~0.051)
#     HIGH (d=0.95): 50 failures → rate = 50/600 = 0.083 Hz  (lerp gives ~0.083)
#   Hard endpoint stretched 30% beyond the original d=1 calibration value of 0.087:
#     d=1 → 0.087 * 1.3 = 0.113 → ~67.8 failures/300s
_SYSMON_RATE_EASY_HZ: float = 0.008  # ~4.8 failures/300 s (total) at d=0
_SYSMON_RATE_HARD_HZ: float = 0.113  # 0.087 * 1.3; ~67.8 total failures/300 s at d=1

# ResMan pump failure rate – TOTAL rate across all 8 pump channels.
#   build_standard_generators divides equally: each pump gets rate / 8.
#   Total events per 300s = rate * 300.
#   Calibrated against pilot scenario files:
#     LOW  (d=0.20):  2 failures → rate =  2/300 = 0.007 Hz  (lerp gives ~0.010)
#     MOD  (d=0.55):  6 failures → rate =  6/300 = 0.020 Hz  (lerp gives ~0.024)
#     HIGH (d=0.95): 12 failures → rate = 12/300 = 0.040 Hz  (lerp gives ~0.039)
#   Hard endpoint stretched 30% beyond the original d=1 calibration value of 0.040:
#     d=1 → 0.040 * 1.3 = 0.052 → ~15.6 pump failures/300s
_RESMAN_PUMP_RATE_EASY_HZ: float = 0.003   # ~0.9 failures/300 s at d=0
_RESMAN_PUMP_RATE_HARD_HZ: float = 0.052   # 0.040 * 1.3; ~15.6 total pump failures/300 s at d=1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _lerp(easy: float, hard: float, d: float) -> float:
    """Linear interpolation between *easy* (d=0) and *hard* (d=1)."""
    return easy + (hard - easy) * d


def _resman_leak(d: float) -> int:
    """Replicate the two-step offset formula from generate_pilot_scenarios.py.

    Examples
    --------
    d=0.20 (LOW)      → 240 ml/min
    d=0.55 (MODERATE) → 560 ml/min
    d=0.95 (HIGH)     → 940 ml/min
    """
    leak = int(_RESMAN_MAX_LEAK_PER_MIN * d)
    if d >= _RESMAN_OFFSET_D_THRESHOLD:
        leak += _RESMAN_OFFSET_MODERATE
    if d >= _RESMAN_OFFSET_HIGH_THRESHOLD:
        leak += _RESMAN_OFFSET_HIGH
    return max(0, leak)


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


def _make_params(d: float) -> TaskParams:
    return TaskParams(
        track_update_ms=_lerp(_TRACK_UPDATE_EASY_MS, _TRACK_UPDATE_HARD_MS, d),
        track_joystick_force=_lerp(_TRACK_FORCE_EASY, _TRACK_FORCE_HARD, d),
        resman_loss_a_per_min=_resman_leak(d),
        resman_loss_b_per_min=_resman_leak(d),
        comms_rate_hz=_lerp(_COMMS_RATE_EASY_HZ, _COMMS_RATE_HARD_HZ, d),
        sysmon_light_rate_hz=_lerp(_SYSMON_RATE_EASY_HZ, _SYSMON_RATE_HARD_HZ, d),
        sysmon_scale_rate_hz=_lerp(_SYSMON_RATE_EASY_HZ, _SYSMON_RATE_HARD_HZ, d),
        resman_pump_rate_hz=_lerp(
            _RESMAN_PUMP_RATE_EASY_HZ, _RESMAN_PUMP_RATE_HARD_HZ, d
        ),
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
        if not (0.0 <= d_min <= d_max <= 1.0):
            raise ValueError(
                f"Invalid difficulty bounds: d_min={d_min}, d_max={d_max}"
            )

        self.d_min = d_min
        self.d_max = d_max
        self.seed = seed

        # Initialised in update()
        self.d: float = 0.0
        self.d_prev: float = 0.0
        self.params: TaskParams = _make_params(0.5)

        self.update(d_init)

    # ------------------------------------------------------------------

    def update(self, d_new: float) -> None:
        """Set a new difficulty level, clamped to [d_min, d_max].

        Stores the previous value in ``d_prev`` for logging delta.
        """
        self.d_prev = self.d
        self.d = max(self.d_min, min(self.d_max, float(d_new)))
        self.params = _make_params(self.d)

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
