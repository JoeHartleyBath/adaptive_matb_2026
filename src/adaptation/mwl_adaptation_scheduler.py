"""mwl_adaptation_scheduler.py

MwlAdaptationScheduler: OpenMATB Scheduler subclass for MWL-driven
tracking-task assistance.

╔══════════════════════════════════════════════════════════════════════╗
║  DESIGN: BINARY TRACKING-ONLY ASSISTANCE TOGGLE                   ║
║                                                                    ║
║  Adaptation is a BINARY ON / OFF toggle applied ONLY to the        ║
║  tracking task.  No other task (comms, sysmon, resman) is touched. ║
║                                                                    ║
║  • Overload detected  (MWL > threshold) → assistance ON            ║
║    Tracking becomes easier by reducing difficulty by the            ║
║    HIGH → MODERATE calibration delta (Δd = 0.30 by default).       ║
║                                                                    ║
║  • Overload resolved  (MWL < threshold) → assistance OFF           ║
║    Tracking returns to its baseline (scenario-set) difficulty.      ║
║                                                                    ║
║  This is NOT graduated multi-task difficulty adjustment.            ║
╚══════════════════════════════════════════════════════════════════════╝

⚠  This module REQUIRES vendor imports (pyglet, core.*) and can ONLY be
   imported inside the OpenMATB bootstrap subprocess.  It is intentionally
   NOT imported by the adaptation package __init__.py.

How it is loaded
----------------
run_openmatb.py injects the following into the bootstrap string when
MWL adaptation mode is active::

    sys.path.insert(0, REPO_SRC_PYTHON)
    from adaptation.mwl_adaptation_scheduler import (
        MwlAdaptationScheduler, MwlAdaptationConfig,
    )
    _cfg = MwlAdaptationConfig(...)
    MwlAdaptationScheduler._MWL_CFG = _cfg
    import core.scheduler as _sched_mod
    _sched_mod.Scheduler = MwlAdaptationScheduler

Architecture
------------
On each pyglet frame::

    super().update(dt)             — normal OpenMATB frame (scenario events
                                     for comms/sysmon/resman run unmodified)
    _pull_mwl(t)                   — drain LSL inlet, feed smoother
    _update_zone(t)                — classify smoothed MWL into above/below/dead
    _run_mwl_policy(t)             — toggle tracking assistance at ~1 Hz
       ├─ _actuate_tracking()      — set tracking params (baseline or assisted)
       └─ _log_mwl_event()         — write JSON to OpenMATB CSV

The policy applies EMA smoothing (α from sweep), a fixed threshold with
hysteresis, a hold timer (MWL must persist for t_hold_s before acting),
and a cooldown between successive toggles.

Tracking assistance
-------------------
MWL > threshold + hysteresis  →  overloaded   →  assistance ON
MWL < threshold − hysteresis  →  not overloaded  →  assistance OFF

When assistance is ON, tracking parameters (taskupdatetime and
joystickforce) are shifted toward easier values by the same magnitude
as the HIGH → MODERATE calibration step (Δd = 0.30 by default).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Optional

import pylsl

from core.scheduler import Scheduler
from core.logger import logger

from adaptation.mwl_smoother import EmaSmoother
from adaptation.adaptation_logger import AdaptationLogger

# Tracking-parameter constants — must stay in sync with difficulty_state.py.
from adaptation.difficulty_state import (
    _TRACK_UPDATE_EASY_MS,
    _TRACK_UPDATE_HARD_MS,
    _TRACK_FORCE_EASY,
    _TRACK_FORCE_HARD,
    _lerp,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MwlAdaptationConfig:
    """All tunable parameters for an MWL-driven adaptation session.

    Defaults match the best sweep configuration (2026-03-16):
    EMA α=0.05, hysteresis 0.02 → 80.3 % BA ± 15.6 %.

    In live sessions ``threshold`` is set per-participant from the Youden J
    value derived from their 2 × 9-min calibration runs (stored in
    ``model_config.json`` by ``calibrate_participant.py``).  The
    default of 0.50 is a code-level fallback used only for desk testing
    without calibration data.

    ⚠  Adaptation is a BINARY tracking-only toggle.  Only the tracking
    task is affected; all other tasks run at their scenario-set levels.
    """

    # Tracking assistance magnitude (in d-space).
    # Default 0.30 = the delta between HIGH (d≈0.60) and MODERATE (d≈0.30)
    # calibration levels.
    # Maps to: taskupdatetime += 12 ms (slower cursor = easier),
    #          joystickforce  += 0.6  (stronger correction = easier).
    assistance_d_delta: float = 0.30

    # Baseline difficulty level.  Set from staircase calibration output.
    # Used to compute baseline and assisted tracking parameters.
    baseline_d: float = 0.50

    # MWL policy (from sweep: α=0.05, hysteresis 0.02)
    # threshold is set per-participant from Youden J (model_config.json);
    # 0.50 is a code-level fallback for desk testing only.
    smoother_alpha: float = 0.05
    threshold: float = 0.50
    hysteresis: float = 0.02
    t_hold_s: float = 3.0        # seconds above/below threshold before toggling
    cooldown_s: float = 15.0     # minimum gap between toggles

    # Signal quality
    signal_quality_min: float = 0.5

    # MWL inlet
    mwl_stream_type: str = "MWL"
    mwl_timeout_s: float = 2.0        # hold if no sample for this long
    mwl_resolve_retry_s: float = 5.0  # retry stream resolution interval

    # Decision tick rate
    decision_interval_s: float = 1.0  # ~1 Hz policy evaluation

    # Audit logger
    audit_csv_path: str = ""  # set by bootstrap; empty = no audit logging

    # Seed (for reproducibility logging)
    seed: int = 0


# ---------------------------------------------------------------------------
# MwlAdaptationScheduler
# ---------------------------------------------------------------------------

class MwlAdaptationScheduler(Scheduler):
    """Scheduler subclass that toggles tracking-task assistance from MWL.

    ⚠  Adaptation is a BINARY ON/OFF toggle applied ONLY to the tracking
    task.  No other task is touched.  This is NOT graduated multi-task
    difficulty adjustment.

    MWL scores are consumed from an external LSL stream (type ``'MWL'``)
    produced by either the real MWL estimator or the simulated source.
    """

    def __init__(self, config: Optional[MwlAdaptationConfig] = None) -> None:
        self._adapt_cfg: MwlAdaptationConfig = (
            config
            or getattr(self.__class__, "_MWL_CFG", None)
            or MwlAdaptationConfig()
        )
        self._adaptation_ready: bool = False
        # super().__init__() is blocking — it drives the pyglet event loop.
        super().__init__()

    # ------------------------------------------------------------------
    # Initialisation (called from first update frame)
    # ------------------------------------------------------------------

    def _setup_adaptation(self) -> None:
        cfg = self._adapt_cfg

        # ── Tracking parameters ──────────────────────────────────────
        # Compute baseline (scenario-set) and assisted tracking params
        # from the calibrated difficulty level and the assistance delta.
        d_assisted = max(0.0, cfg.baseline_d - cfg.assistance_d_delta)

        self._baseline_update_ms: float = _lerp(
            _TRACK_UPDATE_EASY_MS, _TRACK_UPDATE_HARD_MS, cfg.baseline_d,
        )
        self._baseline_force: float = _lerp(
            _TRACK_FORCE_EASY, _TRACK_FORCE_HARD, cfg.baseline_d,
        )
        self._assisted_update_ms: float = _lerp(
            _TRACK_UPDATE_EASY_MS, _TRACK_UPDATE_HARD_MS, d_assisted,
        )
        self._assisted_force: float = _lerp(
            _TRACK_FORCE_EASY, _TRACK_FORCE_HARD, d_assisted,
        )

        # ── Assistance state ─────────────────────────────────────────
        self._assistance_on: bool = False

        # ── MWL smoother ─────────────────────────────────────────────
        self._smoother = EmaSmoother(alpha=cfg.smoother_alpha)

        # ── MWL tracking state ───────────────────────────────────────
        self._mwl_raw: Optional[float] = None
        self._mwl_smoothed: Optional[float] = None
        self._signal_quality: float = 0.0
        self._last_mwl_t: float = 0.0

        # Zone tracking: "above" | "below" | "dead" | None
        self._zone: Optional[str] = None
        self._zone_entry_t: float = 0.0

        # Cooldown
        self._cooldown_end_t: float = 0.0

        # Decision tick
        self._last_decision_t: float = 0.0

        # ── Resolve MWL LSL stream ───────────────────────────────────
        self._mwl_inlet: Optional[pylsl.StreamInlet] = None
        self._last_resolve_t: float = -999.0
        self._resolve_mwl_stream(timeout=5.0)

        # ── Audit logger ─────────────────────────────────────────────
        self._audit_logger: Optional[AdaptationLogger] = None
        if cfg.audit_csv_path:
            self._audit_logger = AdaptationLogger(cfg.audit_csv_path)
            self._audit_logger.open()

        # ── Apply baseline tracking params ───────────────────────────
        self._actuate_tracking()

        self._adaptation_ready = True

        t0 = self.scenario_time
        print(
            f"[MWL-ADAPT] Initialised at t={t0:.2f}s  "
            f"baseline_d={cfg.baseline_d}  assist_delta={cfg.assistance_d_delta}  "
            f"\u03b1={cfg.smoother_alpha}  "
            f"threshold={cfg.threshold}\u00b1{cfg.hysteresis}  "
            f"hold={cfg.t_hold_s}s  cooldown={cfg.cooldown_s}s",
            flush=True,
        )
        print(
            f"[MWL-ADAPT] Tracking params:  "
            f"baseline(update={self._baseline_update_ms:.1f}ms, "
            f"force={self._baseline_force:.2f})  "
            f"assisted(update={self._assisted_update_ms:.1f}ms, "
            f"force={self._assisted_force:.2f})",
            flush=True,
        )

        logger.log_manual_entry(
            json.dumps({
                "event": "mwl_adaptation_init",
                "config": {
                    "baseline_d": cfg.baseline_d,
                    "assistance_d_delta": cfg.assistance_d_delta,
                    "smoother_alpha": cfg.smoother_alpha,
                    "threshold": cfg.threshold,
                    "hysteresis": cfg.hysteresis,
                    "t_hold_s": cfg.t_hold_s,
                    "cooldown_s": cfg.cooldown_s,
                    "signal_quality_min": cfg.signal_quality_min,
                },
                "baseline_tracking": {
                    "taskupdatetime": round(self._baseline_update_ms, 2),
                    "joystickforce": round(self._baseline_force, 3),
                },
                "assisted_tracking": {
                    "taskupdatetime": round(self._assisted_update_ms, 2),
                    "joystickforce": round(self._assisted_force, 3),
                },
            }),
            key="mwl_adaptation",
        )

    def _resolve_mwl_stream(self, timeout: float = 0.5) -> None:
        """Attempt to resolve the MWL LSL stream."""
        self._last_resolve_t = self.scenario_time
        streams = pylsl.resolve_byprop(
            "type", self._adapt_cfg.mwl_stream_type, timeout=timeout,
        )
        if streams:
            self._mwl_inlet = pylsl.StreamInlet(streams[0])
            print(
                f"[MWL-ADAPT] Connected to MWL stream: "
                f"{streams[0].name()} ({streams[0].type()})",
                flush=True,
            )
        else:
            print(
                f"[MWL-ADAPT] WARNING: No MWL stream found "
                f"(type='{self._adapt_cfg.mwl_stream_type}'). "
                f"Will retry every {self._adapt_cfg.mwl_resolve_retry_s}s.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Main loop override
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        super().update(dt)

        if not self._adaptation_ready:
            try:
                self._setup_adaptation()
            except Exception:
                print("[MWL-ADAPT] ERROR during _setup_adaptation:", flush=True)
                traceback.print_exc()
            return

        t = self.scenario_time

        if not self.get_active_plugins():
            # No active plugins yet (scenario just starting) or scenario ended.
            # Only close the audit logger if we are well past t=0 (i.e. scenario
            # has been running for a while and plugins have genuinely all stopped).
            # Never close it in the first 5s — plugins may not have started yet.
            if t > 5.0 and self._audit_logger is not None:
                self._audit_logger.close()
                self._audit_logger = None
            return

        try:
            # Retry stream resolution if needed
            if self._mwl_inlet is None:
                if (t - self._last_resolve_t) >= self._adapt_cfg.mwl_resolve_retry_s:
                    self._resolve_mwl_stream(timeout=0.5)

            # Pull MWL samples and update zone
            self._pull_mwl(t)
            self._update_zone(t)

            # Policy decision tick (~1 Hz)
            if (t - self._last_decision_t) >= self._adapt_cfg.decision_interval_s:
                self._last_decision_t = t
                self._run_mwl_policy(t)
        except Exception:
            print(f"[MWL-ADAPT] ERROR in update at t={t:.2f}s:", flush=True)
            traceback.print_exc()

    # ------------------------------------------------------------------
    # MWL inlet
    # ------------------------------------------------------------------

    def _pull_mwl(self, t: float) -> None:
        """Drain all pending MWL samples from LSL and feed to smoother."""
        if self._mwl_inlet is None:
            return
        try:
            samples, _ = self._mwl_inlet.pull_chunk(timeout=0.0)
        except Exception:
            self._mwl_inlet = None
            print("[MWL-ADAPT] WARNING: MWL stream disconnected", flush=True)
            return
        if not samples:
            return
        for sample in samples:
            self._mwl_smoothed = self._smoother.update(sample[0])
        # Store latest raw values
        self._mwl_raw = samples[-1][0]
        self._signal_quality = samples[-1][2]
        self._last_mwl_t = t

    # ------------------------------------------------------------------
    # Zone tracking
    # ------------------------------------------------------------------

    def _update_zone(self, t: float) -> None:
        """Classify current smoothed MWL into threshold zones."""
        if self._mwl_smoothed is None:
            return
        cfg = self._adapt_cfg
        if self._mwl_smoothed > cfg.threshold + cfg.hysteresis:
            new_zone = "above"
        elif self._mwl_smoothed < cfg.threshold - cfg.hysteresis:
            new_zone = "below"
        else:
            new_zone = "dead"
        if new_zone != self._zone:
            self._zone = new_zone
            self._zone_entry_t = t

    # ------------------------------------------------------------------
    # Policy — binary tracking assistance toggle
    # ------------------------------------------------------------------

    def _run_mwl_policy(self, t: float) -> None:
        """Toggle tracking assistance based on MWL threshold crossings.

        ⚠  BINARY TOGGLE — only the tracking task is affected.
        assist_on  = tracking made easier (overload detected)
        assist_off = tracking returned to baseline (overload resolved)
        hold       = no change
        """
        cfg = self._adapt_cfg
        action = "hold"
        reason = ""

        hold_s = (t - self._zone_entry_t) if self._zone is not None else 0.0
        cooldown_remaining = max(0.0, self._cooldown_end_t - t)

        # Priority-ordered decision logic
        if self._mwl_smoothed is None:
            reason = "no MWL data received"
        elif t - self._last_mwl_t > cfg.mwl_timeout_s:
            reason = "MWL stream timeout"
        elif self._signal_quality < cfg.signal_quality_min:
            reason = (
                f"signal quality {self._signal_quality:.2f} "
                f"< {cfg.signal_quality_min}"
            )
        elif cooldown_remaining > 0:
            reason = f"in cooldown ({cooldown_remaining:.1f}s remaining)"
        elif (
            self._zone == "above"
            and hold_s >= cfg.t_hold_s
            and not self._assistance_on
        ):
            action = "assist_on"
            self._assistance_on = True
            self._cooldown_end_t = t + cfg.cooldown_s
            self._zone_entry_t = t  # reset hold timer
            reason = f"overloaded for {hold_s:.1f}s"
        elif (
            self._zone == "below"
            and hold_s >= cfg.t_hold_s
            and self._assistance_on
        ):
            action = "assist_off"
            self._assistance_on = False
            self._cooldown_end_t = t + cfg.cooldown_s
            self._zone_entry_t = t
            reason = f"not overloaded for {hold_s:.1f}s"
        elif self._zone == "dead":
            reason = "in dead zone"
        elif self._zone == "above" and self._assistance_on:
            reason = "assistance already on"
        elif self._zone == "below" and not self._assistance_on:
            reason = "assistance already off"
        else:
            reason = f"hold timer {hold_s:.1f}s / {cfg.t_hold_s}s"

        # Actuate and log on toggle
        if action != "hold":
            self._actuate_tracking()
            self._log_mwl_event(t, action, reason)
            label = "ON " if action == "assist_on" else "OFF"
            print(
                f"[MWL-ADAPT t={t:6.1f}s] ASSIST {label}  "
                f"smoothed={self._mwl_smoothed:.3f}  {reason}",
                flush=True,
            )

        # Recompute cooldown after potential action for audit log
        cooldown_remaining = max(0.0, self._cooldown_end_t - t)

        # Audit log every decision tick
        if self._audit_logger is not None:
            self._audit_logger.log(
                timestamp_lsl=pylsl.local_clock(),
                scenario_time_s=t,
                mwl_raw=self._mwl_raw if self._mwl_raw is not None else 0.0,
                mwl_smoothed=(
                    self._mwl_smoothed if self._mwl_smoothed is not None else 0.0
                ),
                signal_quality=self._signal_quality,
                threshold=cfg.threshold,
                action=action,
                assistance_on=self._assistance_on,
                cooldown_remaining_s=cooldown_remaining,
                hold_counter_s=hold_s,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Actuation — tracking task ONLY
    # ------------------------------------------------------------------

    def _actuate_tracking(self) -> None:
        """Set tracking plugin parameters to baseline or assisted values.

        ⚠  Only the tracking task is affected.  Comms, sysmon, and resman
        run at their scenario-set levels and are NEVER modified here.
        """
        if "track" not in self.plugins:
            return
        if self._assistance_on:
            self.plugins["track"].set_parameter(
                "taskupdatetime", int(round(self._assisted_update_ms)),
            )
            self.plugins["track"].set_parameter(
                "joystickforce", float(self._assisted_force),
            )
        else:
            self.plugins["track"].set_parameter(
                "taskupdatetime", int(round(self._baseline_update_ms)),
            )
            self.plugins["track"].set_parameter(
                "joystickforce", float(self._baseline_force),
            )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_mwl_event(self, t: float, action: str, reason: str) -> None:
        """Write an assistance-toggle event to the OpenMATB session CSV."""
        payload = {
            "event": "mwl_adaptation_step",
            "t": round(t, 3),
            "action": action,
            "assistance_on": self._assistance_on,
            "mwl_smoothed": (
                round(self._mwl_smoothed, 6)
                if self._mwl_smoothed is not None
                else None
            ),
            "mwl_raw": (
                round(self._mwl_raw, 6)
                if self._mwl_raw is not None
                else None
            ),
            "signal_quality": round(self._signal_quality, 3),
            "reason": reason,
        }
        logger.log_manual_entry(json.dumps(payload), key="mwl_adaptation")
