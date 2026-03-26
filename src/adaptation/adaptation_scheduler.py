"""adaptation_scheduler.py

AdaptationScheduler: OpenMATB Scheduler subclass for online staircase
calibration.

⚠  This module REQUIRES vendor imports (pyglet, core.*) and can ONLY be
   imported inside the OpenMATB bootstrap subprocess.  It is intentionally
   NOT imported by the adaptation package __init__.py.

How it is loaded
----------------
run_openmatb.py injects the following into the bootstrap string when
adaptation mode is active::

    sys.path.insert(0, REPO_SRC_PYTHON)
    from adaptation.adaptation_scheduler import AdaptationScheduler, AdaptationConfig
    import core.scheduler as _sched_mod
    _sched_mod.Scheduler = AdaptationScheduler

which causes OpenMATB's main.py → OpenMATB() → Scheduler() to instantiate
AdaptationScheduler instead.

Session exit
------------
Generator events are suppressed once all plugins are inactive so that the
normal check_if_must_exit() path can terminate the pyglet loop cleanly.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Deque, Dict, List, Optional

# Vendor imports — valid only inside the bootstrap subprocess
from core.scheduler import Scheduler
from core.event import Event
from core.logger import logger

from adaptation.difficulty_state import DifficultyState
from adaptation.staircase_controller import StaircaseController
from adaptation.event_generators import (
    PoissonEventGenerator,
    ScheduledEvent,
    build_standard_generators,
)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptationConfig:
    """All tunable parameters for a staircase calibration session.

    Defaults are conservative values suitable for a 5-minute calibration
    block.  Pass a populated instance to AdaptationScheduler via the
    bootstrap string to override.
    """

    # Difficulty
    d_init: float = 0.5   # start at midpoint so staircase can go up or down
    d_min: float = 0.0
    d_max: float = 1.0
    seed: int = 0

    # Staircase
    target_score: float = 0.70   # target proportion of tracking time in-target
    tolerance: float = 0.05
    window_sec: float = 45.0
    min_samples: int = 5
    # Graduated step schedule: coarse → fine, advancing on each direction reversal.
    # The final entry is the minimum fine step; convergence is declared once
    # stable_ticks_required consecutive no-step ticks occur at that finest step.
    step_schedule: tuple = (0.2, 0.1, 0.05)
    stable_ticks_required: int = 3
    cooldown_sec: float = 20.0

    # Actuation flags (set False to freeze individual task parameters)
    actuate_tracking: bool = True
    actuate_resman: bool = True
    actuate_generators: bool = True

    # How often (scenario seconds) to run the staircase tick
    adaptation_check_interval_sec: float = 5.0

    # Minimum gap (seconds) between any two comms prompts to prevent audio overlap.
    # Audio prompts average ~18 s; the scenario generator uses 18+1=19 s as its
    # minimum prompt-to-prompt spacing.  Use 20 s here to match that with margin.
    min_comms_gap_sec: float = 20.0

    # Maximum cursor deviation (pixels) used to normalise the RMSE score to [0, 1].
    # This is overridden at runtime with the actual track plugin xgain value
    # (= reticle_container.w * 0.4) so that the score range is display-agnostic.
    # This value is only used as a fallback if the track plugin is unavailable.
    max_deviation: float = 200.0


# ---------------------------------------------------------------------------
# Internal performance sample
# ---------------------------------------------------------------------------

@dataclass
class _PerfSample:
    t: float
    module: str
    metric: str
    value: float


# ---------------------------------------------------------------------------
# Line-id counter for injected events (avoids collisions with parsed lines)
# ---------------------------------------------------------------------------
_INJECTED_LINE_ID_START = 90_000


# ---------------------------------------------------------------------------
# AdaptationScheduler
# ---------------------------------------------------------------------------

class AdaptationScheduler(Scheduler):
    """Scheduler subclass that adds online staircase difficulty adaptation.

    Architecture after each pyglet frame
    -------------------------------------
    1. super().update(dt)           — normal OpenMATB frame (events, plugins)
    2. _fire_generator_events(t)    — append due Poisson events to self.events
    3. _run_staircase(t)            — evaluate window; step difficulty if needed
       └─ _actuate()                — set_parameter on all active plugins
       └─ _log_adaptation(...)      — write JSON row with key='adaptation'
    """

    def __init__(self, config: Optional[AdaptationConfig] = None) -> None:
        # Resolve config: explicit arg > class-level _ADAPT_CFG set by bootstrap > defaults
        self._adapt_cfg: AdaptationConfig = (
            config
            or getattr(self.__class__, "_ADAPT_CFG", None)
            or AdaptationConfig()
        )
        self._injected_line_id: int = _INJECTED_LINE_ID_START
        self._adaptation_ready: bool = False
        # super().__init__() is blocking: it calls event_loop.run() and never
        # returns until the session ends.  _setup_adaptation() is therefore
        # called from update() on the very first frame instead.
        super().__init__()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup_adaptation(self) -> None:
        cfg = self._adapt_cfg

        self.state = DifficultyState(
            d_init=cfg.d_init,
            d_min=cfg.d_min,
            d_max=cfg.d_max,
            seed=cfg.seed,
        )

        self.controller = StaircaseController(
            target_score=cfg.target_score,
            tolerance=cfg.tolerance,
            window_sec=cfg.window_sec,
            min_samples=cfg.min_samples,
            step_schedule=cfg.step_schedule,
            cooldown_sec=cfg.cooldown_sec,
            stable_ticks_required=cfg.stable_ticks_required,
        )

        p = self.state.params
        self._generators: Dict[str, PoissonEventGenerator] = build_standard_generators(
            initial_comms_rate_hz=p.comms_rate_hz,
            initial_sysmon_light_rate_hz=p.sysmon_light_rate_hz,
            initial_sysmon_scale_rate_hz=p.sysmon_scale_rate_hz,
            initial_pump_rate_hz=p.resman_pump_rate_hz,
            base_seed=cfg.seed,
        )

        # Performance ring buffer populated via patched logger
        self._perf_buffer: Deque[_PerfSample] = deque()
        self._last_check_t: float = 0.0
        # Refractory tracker for comms: prevents audio overlap
        self._last_comms_t: float = -999.0

        # Intercept logger.log_performance to populate our ring buffer
        # without touching the CSV output path.
        _original_log_perf = logger.log_performance
        _buf = self._perf_buffer
        _self = self   # closure reference

        def _patched_log_performance(module, metric, value):
            _original_log_perf(module, metric, value)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return  # skip non-numeric log entries (e.g. radio names)
            _buf.append(
                _PerfSample(
                    t=_self.scenario_time,
                    module=module,
                    metric=metric,
                    value=numeric_value,
                )
            )

        logger.log_performance = _patched_log_performance

        # Read actual reticle half-width from the track plugin geometry.
        # Track.create_widgets() sets xgain = reticle_container.w * gain_ratio / 2
        # (gain_ratio = 0.8), making xgain the maximum possible cursor displacement.
        # Using this as max_deviation means score = 1 - RMSE/xgain spans the full
        # [0, 1] range on this display, rather than the static 200 px fallback.
        if "track" in self.plugins and hasattr(self.plugins["track"], "xgain"):
            self._adapt_cfg.max_deviation = self.plugins["track"].xgain
            print(
                f"[ADAPTATION] max_deviation set from track geometry: "
                f"{self._adapt_cfg.max_deviation:.1f} px",
                flush=True,
            )

        # Apply d_init parameters to plugins immediately
        self._actuate()

        # Initialise generators relative to current scenario time so first
        # events fire ~1/rate seconds from now (not from t=0).
        t0 = self.scenario_time
        for gen in self._generators.values():
            gen.begin(start_t=t0)

        # All attributes ready — unlock the update() loop.
        self._adaptation_ready = True

        print(
            f"[ADAPTATION] Initialised at t={t0:.2f}s  "
            f"d={cfg.d_init}  target={cfg.target_score}  "
            f"window={cfg.window_sec}s  schedule={list(cfg.step_schedule)}  "
            f"stable_ticks={cfg.stable_ticks_required}",
            flush=True,
        )

        logger.log_manual_entry(
            json.dumps({
                "event": "adaptation_init",
                "config": {
                    "d_init": cfg.d_init,
                    "seed": cfg.seed,
                    "target_score": cfg.target_score,
                    "tolerance": cfg.tolerance,
                    "window_sec": cfg.window_sec,
                    "step_schedule": list(cfg.step_schedule),
                    "stable_ticks_required": cfg.stable_ticks_required,
                    "cooldown_sec": cfg.cooldown_sec,
                    "max_deviation": cfg.max_deviation,
                },
                "initial_params": self.state.as_dict(),
            }),
            key="adaptation",
        )

    # ------------------------------------------------------------------
    # Main loop override
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        # Run the normal OpenMATB frame first (timers, plugins, events)
        super().update(dt)

        # First-frame setup: super().__init__() blocks in event_loop.run(), so
        # _setup_adaptation() must be called from inside the running event loop.
        if not self._adaptation_ready:
            try:
                self._setup_adaptation()
            except Exception:
                print("[ADAPTATION] ERROR during _setup_adaptation:", flush=True)
                traceback.print_exc()
            return  # start adaptation logic from next frame

        t = self.scenario_time

        # Only proceed while at least one plugin is alive; once all plugins
        # have stopped we let check_if_must_exit() do its job cleanly.
        if not self.get_active_plugins():
            return

        try:
            if self._adapt_cfg.actuate_generators:
                self._fire_generator_events(t)

            if (t - self._last_check_t) >= self._adapt_cfg.adaptation_check_interval_sec:
                self._last_check_t = t
                self._run_staircase(t)
        except Exception:
            print(f"[ADAPTATION] ERROR in update at t={t:.2f}s:", flush=True)
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Staircase
    # ------------------------------------------------------------------

    def _run_staircase(self, t: float) -> None:
        score = self._compute_composite_score(t)
        if score is not None:
            self.controller.push_performance(t, score)

        n_buf = sum(
            1 for s in self._perf_buffer
            if s.module == "track" and s.metric == "center_deviation"
        )

        delta = self.controller.tick(t)

        # Convergence: staircase has stabilised at the finest step.
        if self.controller.converged:
            print(
                f"[ADAPTATION t={t:6.1f}s] CONVERGED  "
                f"d={self.state.d:.3f}  Ending block.",
                flush=True,
            )
            self._log_adaptation(t, delta=None, score=score)
            logger.log_manual_entry(
                json.dumps({"event": "adaptation_converged", "t": round(t, 3), "d": self.state.d}),
                key="adaptation",
            )
            try:
                import pyglet
                pyglet.app.exit()
            except Exception:
                pass
            return

        if delta is not None:
            d_old = self.state.d
            self.state.update(self.state.d + delta)

            if abs(self.state.d - d_old) < 1e-9:
                # d didn't move — clamped at ceiling or floor.
                boundary_label = "CEILING" if delta > 0 else "FLOOR"
                self.controller.notify_boundary()
                print(
                    f"[ADAPTATION t={t:6.1f}s] {boundary_label} (clamped)  "
                    f"d={self.state.d:.3f}  score={score:.3f}  "
                    f"boundary_ticks={self.controller._boundary_ticks}/{self.controller.stable_ticks_required}",
                    flush=True,
                )
                # Check immediately: boundary ticks may have just triggered convergence.
                if self.controller.converged:
                    print(
                        f"[ADAPTATION t={t:6.1f}s] CONVERGED (boundary)  "
                        f"d={self.state.d:.3f}  Ending block.",
                        flush=True,
                    )
                    self._log_adaptation(t, delta=None, score=score)
                    logger.log_manual_entry(
                        json.dumps({"event": "adaptation_converged", "t": round(t, 3), "d": self.state.d}),
                        key="adaptation",
                    )
                    try:
                        import pyglet
                        pyglet.app.exit()
                    except Exception:
                        pass
                return

            self._actuate()
            self._log_adaptation(t, delta=delta, score=score)
            direction = "UP  " if delta > 0 else "DOWN"
            step_idx = self.controller._schedule_idx
            step_val = self.controller.step_up
            print(
                f"[ADAPTATION t={t:6.1f}s] STEP {direction}  "
                f"d: {d_old:.3f} -> {self.state.d:.3f}  "
                f"score={score:.3f}  step={step_val:.2f} (schedule[{step_idx}])  buf={n_buf}",
                flush=True,
            )
        else:
            score_str = f"{score:.3f}" if score is not None else "N/A (window filling)"
            stable = self.controller._stable_ticks_at_final_step
            at_final = self.controller._schedule_idx >= len(self.controller._step_schedule) - 1
            stable_str = f"  stable={stable}/{self.controller.stable_ticks_required}" if at_final else ""
            print(
                f"[ADAPTATION t={t:6.1f}s] monitoring  "
                f"d={self.state.d:.3f}  score={score_str}  buf={n_buf}{stable_str}",
                flush=True,
            )

    def _compute_composite_score(self, t: float) -> Optional[float]:
        """Normalised [0, 1] performance score from tracking RMSE.

        Uses `center_deviation` (Euclidean distance from reticle centre, pixels)
        over the evaluation window.  RMSE is normalised by `max_deviation` and
        inverted so that 1.0 = perfect tracking and 0.0 = maximum error.

        Returns None when there are fewer than min_samples valid entries.
        """
        cutoff = t - self._adapt_cfg.window_sec
        # Evict expired samples
        while self._perf_buffer and self._perf_buffer[0].t < cutoff:
            self._perf_buffer.popleft()

        deviation_samples = [
            s.value
            for s in self._perf_buffer
            if s.module == "track" and s.metric == "center_deviation"
        ]
        if len(deviation_samples) < self._adapt_cfg.min_samples:
            return None

        rmse = math.sqrt(sum(v ** 2 for v in deviation_samples) / len(deviation_samples))
        return max(0.0, 1.0 - rmse / self._adapt_cfg.max_deviation)

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def _actuate(self) -> None:
        """Push current DifficultyState parameters into active plugins."""
        p = self.state.params
        cfg = self._adapt_cfg

        if cfg.actuate_tracking and "track" in self.plugins:
            self.plugins["track"].set_parameter(
                "taskupdatetime", int(round(p.track_update_ms))
            )
            # set_parameter writes directly to the parameters dict without running
            # validation, so floats are accepted here (no need for int conversion).
            # joystickforce range: 3.0 (d=0, easy) → 1.0 (d=1, hard).
            # 1.0 is the theoretical floor: at full joystick deflection (±1.0)
            # the participant can just barely cancel peak sinusoidal drift.
            self.plugins["track"].set_parameter(
                "joystickforce", float(p.track_joystick_force)
            )

        if cfg.actuate_resman and "resman" in self.plugins:
            self.plugins["resman"].set_parameter(
                "tank-a-lossperminute", p.resman_loss_a_per_min
            )
            self.plugins["resman"].set_parameter(
                "tank-b-lossperminute", p.resman_loss_b_per_min
            )

        if cfg.actuate_generators:
            self._update_generator_rates()

    def _update_generator_rates(self) -> None:
        """Propagate current DifficultyState event rates to all generators."""
        p = self.state.params
        for key, gen in self._generators.items():
            if key.startswith("comms_"):
                gen.update_rate(max(1e-6, p.comms_rate_hz * 0.5))
            elif key.startswith("sysmon_light_"):
                gen.update_rate(max(1e-6, p.sysmon_light_rate_hz / 2))
            elif key.startswith("sysmon_scale_"):
                gen.update_rate(max(1e-6, p.sysmon_scale_rate_hz / 4))
            elif key.startswith("resman_pump_"):
                gen.update_rate(max(1e-6, p.resman_pump_rate_hz / 8))

    # ------------------------------------------------------------------
    # Generator event firing
    # ------------------------------------------------------------------

    def _fire_generator_events(self, t: float) -> None:
        for key, gen in self._generators.items():
            while gen.ready(t):
                # Enforce minimum inter-comms gap to prevent audio overlap.
                # Do NOT pop the event yet — defer until the gap clears.
                if key.startswith("comms_"):
                    if t - self._last_comms_t < self._adapt_cfg.min_comms_gap_sec:
                        break

                scheduled_events: List[ScheduledEvent] = gen.pop(t)
                for sched in scheduled_events:
                    if sched.plugin in self.plugins:
                        self.events.append(self._to_vendor_event(sched))

                if key.startswith("comms_"):
                    self._last_comms_t = t

    def _to_vendor_event(self, sched: ScheduledEvent) -> Event:
        """Convert a ScheduledEvent to a vendor core.event.Event."""
        self._injected_line_id += 1
        return Event(
            self._injected_line_id,
            sched.time_sec,
            sched.plugin,
            sched.command,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_adaptation(
        self,
        t: float,
        *,
        delta: Optional[float],
        score: Optional[float],
    ) -> None:
        """Write a structured adaptation step row to the session CSV."""
        payload = {
            "event": "adaptation_step",
            "t": round(t, 3),
            "delta": round(delta, 6) if delta is not None else None,
            "score": round(score, 4) if score is not None else None,
            "state": self.state.as_dict(),
            "controller": self.controller.as_dict(),
        }
        logger.log_manual_entry(json.dumps(payload), key="adaptation")
