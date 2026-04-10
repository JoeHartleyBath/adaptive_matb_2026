"""End-to-end orchestrator for a single participant study session.

Automates the full flow:
  1. Practice familiarisation (4 scenarios — no sensors required)
  2. Staircase calibration (adaptation_skeleton.txt — no sensors required)
     — sensor setup pause —
  3. Rest baseline (2-min fixation cross for EEG normalisation)
  4. Scenario generation (calibration + adaptation scenarios from d_final)
  5. Calibration runs (2 × 9-min counterbalanced scenarios)
  6. Model calibration (scratch 3-class SVM-linear on participant's MATB cal data)
  7. Adaptation condition (generated scenario + MWL-driven toggle)
     / Control condition (same scenario, no adaptation)
     — conditions 7 counterbalanced via adaptation_first in config
  8. Post-session verification + analysis + plots

Model strategy (validated 2026-04-02, n=1 PSELF pilot):
  A scratch 3-class SVM-linear (SelectKBest k=35 + StandardScaler + SVC linear
  C=1.0) trained directly on the participant's MATB calibration data
  outperforms TSST-group warm-start on every metric including within-block
  performance correlation.  No group model needed.

  LogReg and SVM-RBF were also tested (sweep_scratch_models.py); SVM-linear
  selected after sweep on 2026-04-02 (see lab notes 2026-04-02_pself_run.md).
  Linear SVM is comparable (AUCbin 0.743 vs 0.713) but P(H)|LOW=0.68 would
  saturate the adaptation scheduler at baseline.  SVM-RBF overfits the short
  cal blocks and collapses on adaptation windows.  LR_C0.003 retained.

Usage:
    python src/run_full_study_session.py --participant P001 \\
        --output-root C:/adaptive_matb_data

The script pauses for operator confirmation between major phases.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSIGNMENTS_PATH = _REPO_ROOT / "config" / "participant_assignments.yaml"
_SCENARIOS_DIR = _REPO_ROOT / "experiment" / "scenarios"

# ---------------------------------------------------------------------------
# Venv guard
# ---------------------------------------------------------------------------

def _check_venv() -> None:
    """Warn if not running inside the project virtual environment.

    Subprocesses (HR streamer, EDA streamer, etc.) inherit sys.executable, so
    running with the wrong Python silently breaks them.
    """
    expected_venv = _REPO_ROOT / ".venv"
    current_python = Path(sys.executable).resolve()
    try:
        current_python.relative_to(expected_venv.resolve())
    except ValueError:
        print(
            f"\n{'!'*60}\n"
            f"  WARNING: Not running inside the project venv.\n"
            f"  Active Python : {sys.executable}\n"
            f"  Expected venv : {expected_venv}\n"
            f"\n"
            f"  Sensor streamers (HR, EDA) will likely fail.\n"
            f"  Run with:\n"
            f"    {expected_venv / 'Scripts' / 'python.exe'} src/run_full_study_session.py ...\n"
            f"  or activate the venv first:\n"
            f"    {expected_venv / 'Scripts' / 'Activate.ps1'}\n"
            f"{'!'*60}\n",
            file=sys.stderr,
        )
        if not _NO_PAUSE:
            input("  Press ENTER to continue anyway, or Ctrl-C to abort... ")
        else:
            print("  (--no-pause: auto-continuing)")
        print()

# Default output root matches run_openmatb.py convention.
_DEFAULT_OUTPUT_ROOT = Path(r"C:\data\adaptive_matb")
_LABRECORDER_EXE = Path(r"C:\LabRecorder\LabRecorder.exe")


def _ensure_labrecorder_running(host: str = "127.0.0.1", port: int = 22345, timeout: float = 15.0) -> None:
    """Launch LabRecorder.exe if its RCS port is not yet listening, then wait."""
    def _rcs_open() -> bool:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            return False

    if _rcs_open():
        print("  LabRecorder RCS: already running.")
        return

    if not _LABRECORDER_EXE.exists():
        print(
            f"  WARNING: LabRecorder not found at {_LABRECORDER_EXE}\n"
            "  Start it manually before recording begins."
        )
        return

    print(f"  Launching LabRecorder ({_LABRECORDER_EXE.name})...")
    subprocess.Popen([str(_LABRECORDER_EXE)], close_fds=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _rcs_open():
            print("  LabRecorder RCS: ready.")
            return
        time.sleep(0.5)

    print(
        f"  WARNING: LabRecorder RCS port {port} did not open within {timeout:.0f}s.\n"
        "  Recording may not start — check LabRecorder manually."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoke_test_dual_eeg_inlet(stream_substr: str = "eego", timeout_s: float = 15.0) -> bool:
    """Quick smoke test: confirm both EEG amps are streaming before phase 6.

    Resolves EEG streams whose name contains *stream_substr*, checks that at
    least 2 are found, pulls one small chunk from each, and prints a summary.
    Aborts early (within ~2 s per attempt) so the session launch is not held
    up for long.

    Returns True if both streams are live and returning data, False otherwise.
    """
    try:
        import pylsl
    except ImportError:
        print("  [EEG smoke test] pylsl not available — skipping.")
        return True  # don't block if pylsl is missing

    print(f"\n  [EEG smoke test] Checking for ≥2 EEG streams containing '{stream_substr}'...", flush=True)
    all_eeg = pylsl.resolve_stream("type", "EEG", 2, timeout_s)
    matched = sorted([s for s in all_eeg if stream_substr in s.name()], key=lambda s: s.name())

    if len(matched) < 2:
        print(
            f"  [EEG smoke test] FAIL — found {len(matched)} stream(s), need 2.\n"
            "  Are both amplifiers powered on and streaming?\n"
            "  Check eego software or restart the amps before proceeding.",
            flush=True,
        )
        return False

    # Pull a tiny chunk from each to confirm live data
    all_ok = True
    for s in matched[:2]:
        inlet = pylsl.StreamInlet(s)
        samples, _ = inlet.pull_chunk(timeout=2.0, max_samples=64)
        n = len(samples)
        ch = s.channel_count()
        srate = s.nominal_srate()
        status = "OK" if n > 0 else "NO DATA"
        print(
            f"  [EEG smoke test] {status:7s} {s.name():50s}  {ch} ch @ {srate:.0f} Hz  ({n} samples pulled)",
            flush=True,
        )
        if n == 0:
            all_ok = False

    if all_ok:
        print("  [EEG smoke test] Both EEG streams live. Proceeding.\n", flush=True)
    else:
        print("  [EEG smoke test] FAIL — one or both streams returned no data.", flush=True)
    return all_ok


def _load_participant_config(pid: str) -> dict:
    """Load participant entry from participant_assignments.yaml."""
    with _ASSIGNMENTS_PATH.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    participants = data.get("participants", {})
    if pid not in participants:
        sys.exit(
            f"ERROR: {pid} not found in {_ASSIGNMENTS_PATH}\n"
            f"Add them manually to {_ASSIGNMENTS_PATH} — see docs/PARTICIPANT_ASSIGNMENTS.md."
        )
    return participants[pid]


def _next_session_id(cfg: dict) -> str:
    """Derive next session ID from sessions_completed list."""
    completed = cfg.get("sessions_completed", [])
    return f"S{len(completed) + 1:03d}"


# Module-level flag set by --no-pause to suppress all interactive prompts.
_NO_PAUSE: bool = False


def _pause(msg: str) -> None:
    """Pause for operator confirmation. Ctrl-C aborts."""
    print(f"\n{'='*60}")
    print(f"  PAUSE: {msg}")
    print(f"{'='*60}")
    if _NO_PAUSE:
        print("  (--no-pause: auto-continuing)")
    else:
        input("  Press ENTER to continue (Ctrl-C to abort)... ")
    print()


def _run(cmd: list[str], label: str, allow_fail: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with clear labelling.

    If allow_fail is False (default), exits the session on non-zero return code.
    If allow_fail is True, prints a warning and returns the result for the caller to handle.
    """
    print(f"\n>>> [{label}]")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        if allow_fail:
            print(f"\n  WARNING: [{label}] exited with code {result.returncode} (continuing).")
        else:
            sys.exit(f"\n!!! [{label}] failed with exit code {result.returncode}")
    else:
        print(f"    [{label}] OK")
    return result


# ---------------------------------------------------------------------------
# Shared run_openmatb invocation builder
# ---------------------------------------------------------------------------

_PRACTICE_SCENARIOS = [
    "pilot_practice_intro.txt",
    "pilot_practice_low.txt",
    "pilot_practice_moderate.txt",
    "pilot_practice_high.txt",
]


def _openmatb_base_cmd(ctx: dict) -> list[str]:
    """Common args for every run_openmatb.py invocation."""
    cmd = [
        ctx["python"],
        str(ctx["repo_root"] / "src" / "run_openmatb.py"),
        "--participant", ctx["pid"],
        "--session", ctx["session"],
        "--output-root", str(ctx["output_root"]),
        "--skip-assignment-update",
        "--pilot1",  # always: dual-amp EEG, physiology recording, XDF QC
    ]
    if ctx["verification"]:
        cmd += ["--verification", "--skip-stream-check", "--speed", str(ctx["speed"])]
    elif ctx.get("skip_stream_check"):
        cmd += ["--skip-stream-check"]
    if ctx["labrecorder_rcs"]:
        cmd += ["--labrecorder-rcs"]
    if ctx.get("eda_auto_port"):
        cmd += ["--eda-auto-port"]
    return cmd


def _find_rest_xdf(ctx: dict, acq: str = "rest") -> Path | None:
    """Return the resting-baseline XDF path written by LabRecorder, or None.

    Searches for the most-recently-modified ``*acq-{acq}*.xdf`` file in the
    BIDS physiology directory.  Pass ``acq='rest_adapt'`` to locate the
    pre-adaptation refresh recording.
    """
    phys_dir = (
        ctx["output_root"] / "physiology"
        / f"sub-{ctx['pid']}" / f"ses-{ctx['session']}" / "physio"
    )
    candidates = sorted(phys_dir.glob(f"*acq-{acq}*.xdf"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return candidates[-1]


def _find_latest_session_csv(ctx: dict) -> Path:
    """Find the most recent session CSV from manifests in the sessions dir."""
    sessions_dir = ctx["session_data_dir"] / "sessions"
    manifests = sorted(sessions_dir.glob("**/*.manifest.json"))
    if not manifests:
        sys.exit(f"ERROR: No manifests found in {sessions_dir}")
    # Use the last manifest (most recent run)
    manifest_path = manifests[-1]
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    csv_path_str = manifest.get("paths", {}).get("session_csv")
    if not csv_path_str:
        sys.exit(f"ERROR: No session_csv path in {manifest_path}")
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        sys.exit(f"ERROR: Session CSV not found: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

def phase_practice(ctx: dict) -> None:
    """Phase 1: Practice familiarisation (4 scenarios — no sensors required)."""
    print("\n" + "=" * 60)
    print("  PHASE 1: PRACTICE")
    print("=" * 60)
    for i, scenario in enumerate(_PRACTICE_SCENARIOS, 1):
        cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", scenario, "--skip-stream-check"]
        _run(cmd, f"Practice {i}/{len(_PRACTICE_SCENARIOS)}: {scenario}")


def phase_rest_baseline(ctx: dict) -> None:
    """Phase 3: 2-min resting-state EEG baseline (fixation cross).

    Runs after sensor placement; the normal preflight inside run_openmatb.py
    acts as the sensor verification gate before recording begins.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: REST BASELINE")
    print("=" * 60)
    _pause(
        "EEG BASELINE — All sensors should now be fitted and checked.\n"
        "  Instruct participant:\n"
        "  'A fixation cross will appear on screen.\n"
        "   Please sit quietly, keep your eyes fixed on the cross,\n"
        "   and try not to blink or move for 2 minutes.'\n"
        "  Press ENTER when participant is settled and ready."
    )
    cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", "rest_baseline.txt"]
    if ctx["labrecorder_rcs"]:
        cmd += ["--labrecorder-acq", "rest"]
    _run(cmd, "Rest baseline (fixation cross)")

    # Store the XDF path for phase_model_calibration to consume.
    # Only resolvable when LabRecorder is active.
    if ctx["labrecorder_rcs"]:
        rest_xdf = _find_rest_xdf(ctx)
        if rest_xdf:
            ctx["rest_baseline_xdf"] = rest_xdf
            print(f"  Rest XDF: {rest_xdf}")
        else:
            print("  WARNING: Rest baseline XDF not found — norm will fall back to LOW block.")


def phase_staircase(ctx: dict) -> None:
    """Phase 2: Staircase calibration -> d_final (sensors not required)."""
    print("\n" + "=" * 60)
    print("  PHASE 2: STAIRCASE CALIBRATION")
    print("=" * 60)
    cmd = _openmatb_base_cmd(ctx) + [
        "--only-scenario", "adaptation_skeleton.txt",
        "--adaptation",
        "--skip-stream-check",  # No sensors fitted at this point
    ]
    # Strip physiology/recording flags — sensors are not fitted during staircase.
    # --pilot1 enforces a physiology artifact check at run end; remove it entirely.
    cmd = [a for a in cmd if a not in ("--labrecorder-rcs", "--pilot1", "--eda-auto-port")]
    result = _run(cmd, "Staircase calibration", allow_fail=True)

    # Extract d_final from the staircase session CSV regardless of exit code.
    # The scenario completes before run_openmatb does its post-run checks, so
    # the CSV is written even when the process exits non-zero.
    try:
        staircase_csv = _find_latest_session_csv(ctx)
        sys.path.insert(0, str(ctx["repo_root"] / "scripts" / "generate_scenarios"))
        from generate_full_study_scenarios import extract_d_final
        d_final = extract_d_final(staircase_csv)
        ctx["d_final"] = d_final
        ctx["staircase_csv"] = staircase_csv
        print(f"\n  d_final = {d_final:.4f}  (from {staircase_csv.name})")
        if result.returncode != 0:
            print(
                f"\n  Staircase scenario completed (d_final={d_final:.4f}) but run_openmatb"
                f" exited with code {result.returncode}.\n"
                f"  This is expected if post-run checks require physiology data.\n"
                f"  You can continue from Phase 3 with:\n"
                f"    python src/run_full_study_session.py --participant {ctx['pid']}"
                f" --labrecorder-rcs --eda-auto-port --post-phase-verify --start-phase 3"
            )
    except Exception as exc:
        print(f"\n  ERROR: Could not extract d_final from staircase CSV: {exc}")
        sys.exit(1)


def phase_generate_scenarios(ctx: dict) -> None:
    """Phase 4: Generate calibration + adaptation scenarios from d_final."""
    print("\n" + "=" * 60)
    print("  PHASE 4: GENERATE SCENARIOS")
    print("=" * 60)
    pid = ctx["pid"]
    d_final = ctx["d_final"]
    py = ctx["python"]
    gen_dir = str(ctx["repo_root"] / "scripts" / "generate_scenarios")

    # Two calibration scenarios (conditions 1 and 2, counterbalanced block order)
    for condition in (1, 2):
        _run(
            [
                py, str(Path(gen_dir) / "generate_full_study_scenarios.py"),
                "--participant", pid,
                "--condition", str(condition),
                "--d-final", f"{d_final:.4f}",
                "--output-dir", str(ctx["scenarios_dir"]),
            ],
            f"Generate calibration scenario (condition {condition})",
        )

    # One adaptation scenario (condition 1; same file used for control)
    _run(
        [
            py, str(Path(gen_dir) / "generate_adaptive_automation_scenarios.py"),
            "--participant", pid,
            "--condition", "1",
            "--d-final", f"{d_final:.4f}",
            "--output-dir", str(ctx["scenarios_dir"]),
        ],
        "Generate adaptation scenario",
    )

    # Store generated filenames in context for later phases
    pid_lower = pid.lower()
    ctx["calibration_scenario_c1"] = f"full_calibration_{pid_lower}_c1.txt"
    ctx["calibration_scenario_c2"] = f"full_calibration_{pid_lower}_c2.txt"
    ctx["adaptation_scenario"] = f"adaptive_automation_{pid_lower}_c1_8min.txt"

    print(f"\n  Calibration C1 : {ctx['calibration_scenario_c1']}")
    print(f"  Calibration C2 : {ctx['calibration_scenario_c2']}")
    print(f"  Adaptation     : {ctx['adaptation_scenario']}")


def phase_calibration_runs(ctx: dict) -> None:
    """Phase 5: Two 9-min calibration runs (counterbalanced)."""
    print("\n" + "=" * 60)
    print("  PHASE 5: CALIBRATION RUNS")
    print("=" * 60)
    for condition in (1, 2):
        scenario = ctx[f"calibration_scenario_c{condition}"]
        cmd = _openmatb_base_cmd(ctx) + [
            "--only-scenario", scenario,
            "--skip-stream-check",
        ]
        if ctx["labrecorder_rcs"]:
            cmd += ["--labrecorder-acq", f"cal_c{condition}"]
        _run(cmd, f"Calibration run C{condition}: {scenario}")
        if condition == 1:
            _pause("Calibration run 1 complete. Collect TLX, then continue.")


def phase_model_calibration(ctx: dict) -> None:
    """Phase 6: Calibrate participant model — scratch 3-class SVM-linear on MATB cal data.

    Trains a fresh SelectKBest(k=35) + StandardScaler + SVC(kernel='linear', C=1.0)
    directly on this participant's MATB calibration XDFs.  No group or pretrain
    model is used.

    Model selection: SVM-linear (k=35, C=1.0) was selected over LogReg and SVM-RBF
    via sweep_scratch_models.py (2026-04-02 PSELF pilot).  SVM-linear avoids the
    P(HIGH)≈0 saturation seen with the earlier LogReg at baseline and gives better
    adaptation-condition separation.  See lab notes 2026-04-02_pself_run.md.
    k reduced from 40 → 35 (2026-04-10, selectk_sweep_s005_block01).
    """
    print("\n" + "=" * 60)
    print("  PHASE 6: MODEL CALIBRATION")
    print("=" * 60)
    pid = ctx["pid"]
    py = ctx["python"]

    # XDF directory: where LabRecorder saved calibration recordings
    xdf_dir = (
        ctx["output_root"] / "physiology"
        / f"sub-{pid}" / f"ses-{ctx['session']}" / "physio"
    )
    model_out = ctx["model_out_dir"]

    calibrate_cmd = [
        py, str(ctx["repo_root"] / "scripts" / "calibrate_participant.py"),
        "calibrate",
        "--scratch",   # 3-class; fits own selector + scaler on this participant's MATB cal data
        "--xdf-dir", str(xdf_dir),
        "--pid", pid,
        "--out-dir", str(model_out),
    ]
    # Resting XDF (Phase 1) provides the outer feature normalisation reference.
    # Even in scratch mode the first normalisation step uses rest vs task contrast;
    # the inner StandardScaler is then fit on those already-normalised features.
    rest_xdf = ctx.get("rest_baseline_xdf")
    if rest_xdf and rest_xdf.exists():
        calibrate_cmd += ["--resting-xdf", str(rest_xdf)]
        print(f"  Resting XDF for norm stats: {rest_xdf.name}")
    else:
        print("  No resting XDF — norm stats will use LOW block fallback.")

    _run(calibrate_cmd, "Calibrate participant model (scratch 3-class)")
    ctx["participant_model_dir"] = model_out

    # Read the Youden J threshold written by the calibration script.
    cfg_path = model_out / "model_config.json"
    mwl_cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    ctx["mwl_threshold"] = mwl_cfg_data["youden_threshold"]

    print(f"\n  Model artefacts: {model_out}")
    print(f"  MWL threshold (Youden J): {ctx['mwl_threshold']:.4f}")


def phase_pre_adaptation_baseline(ctx: dict) -> None:
    """Record a fresh 60-s resting baseline and refresh norm_stats.json.

    Called immediately before the adaptation condition.  Addresses between-session
    EEG non-stationarity: the resting feature distribution shifts by ~30 min even
    within the same lab visit, causing P(HIGH) saturation (see ADR-0005).
    """
    print("\n" + "=" * 60)
    print("  PRE-ADAPTATION BASELINE REFRESH")
    print("=" * 60)
    _pause(
        "EEG BASELINE REFRESH — Instruct participant:\n"
        "  'Before the next task, please sit quietly with your eyes open,\n"
        "   focus on the fixation cross, and try not to move for 1 minute.'\n"
        "  Press ENTER when participant is ready."
    )
    cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", "rest_baseline.txt"]
    if ctx["labrecorder_rcs"]:
        cmd += ["--labrecorder-acq", "rest_adapt"]
    _run(cmd, "Pre-adaptation rest baseline")

    # Locate the fresh XDF and refresh norm_stats.json
    model_dir = ctx.get("participant_model_dir")
    if not (ctx["labrecorder_rcs"] and model_dir):
        print("  Skipping baseline refresh: no LabRecorder or model dir.")
        return

    rest_xdf = _find_rest_xdf(ctx, acq="rest_adapt")
    if not rest_xdf:
        print("  WARNING: Pre-adaptation rest XDF not found — norm_stats.json NOT refreshed.")
        return

    _run(
        [
            ctx["python"],
            str(ctx["repo_root"] / "scripts" / "update_session_baseline.py"),
            "--xdf", str(rest_xdf),
            "--model-dir", str(model_dir),
            "--duration", "60",
        ],
        "Refresh norm_stats.json from pre-adaptation baseline",
    )
    print(f"  norm_stats.json refreshed — MWL estimator will use new baseline.")


def phase_experimental_conditions(ctx: dict) -> None:
    """Phase 7: Adaptation and control conditions (counterbalanced)."""
    print("\n" + "=" * 60)
    print("  PHASE 7: EXPERIMENTAL CONDITIONS")
    print(f"  Order: {ctx['condition_order'][0]} -> {ctx['condition_order'][1]}")
    print("=" * 60)
    scenario = ctx["adaptation_scenario"]

    # Smoke-test both EEG amps before any scenario launch (fast fail)
    if "adaptation" in ctx["condition_order"] and not ctx.get("skip_smoke_test"):
        if not _smoke_test_dual_eeg_inlet():
            sys.exit("Aborted: EEG dual-inlet smoke test failed. Fix EEG streams and re-run.")

    for i, condition in enumerate(ctx["condition_order"], 1):
        label = f"Condition {i}/{len(ctx['condition_order'])}: {condition}"
        cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", scenario, "--skip-stream-check"]

        if condition == "adaptation":
            cmd += [
                "--mwl-adaptation",
                "--mwl-model-dir", str(ctx["participant_model_dir"]),
                "--mwl-audit-csv", str(ctx["audit_csv"]),
            ]
            if ctx.get("mwl_threshold") is not None:
                cmd += ["--mwl-threshold", f"{ctx['mwl_threshold']:.6f}"]
            if ctx.get("d_final") is not None:
                cmd += ["--mwl-baseline-d", f"{ctx['d_final']:.4f}"]
        if ctx["labrecorder_rcs"]:
            cmd += ["--labrecorder-acq", condition]

        _run(cmd, label)

        # Capture session CSV for post-session analysis
        if condition == "adaptation":
            ctx["adaptation_session_csv"] = _find_latest_session_csv(ctx)
        elif condition == "control":
            ctx["control_session_csv"] = _find_latest_session_csv(ctx)

        if i < len(ctx["condition_order"]):
            _pause(f"{condition.title()} condition complete. Collect TLX, then continue.")


def phase_post_session(ctx: dict) -> None:
    """Phase 8: Verification, analysis, plots."""
    print("\n" + "=" * 60)
    print("  PHASE 8: POST-SESSION ANALYSIS")
    print("=" * 60)
    py = ctx["python"]
    repo = ctx["repo_root"]
    audit_csv = ctx["audit_csv"]
    session_csv = ctx.get("adaptation_session_csv")
    pid = ctx["pid"]

    # 8a. Verification (best-effort — don't abort the session on failure)
    if audit_csv.exists():
        verify_cmd = [
            py, str(repo / "tests" / "verification" / "verify_mwl_adaptation_session.py"),
            "--csv", str(audit_csv),
        ]
        print(f"\n>>> [Verification]")
        print(f"    {' '.join(verify_cmd)}")
        result = subprocess.run(verify_cmd)
        if result.returncode != 0:
            print("    WARNING: Verification reported issues (see above).")
        else:
            print("    [Verification] OK")
    else:
        print(f"\n  Skipping verification — audit CSV not found: {audit_csv}")

    # 8b. Analysis
    if audit_csv.exists() and session_csv and session_csv.exists():
        analysis_out = ctx["session_data_dir"] / f"adaptation_analysis_{pid.lower()}.json"
        _run(
            [
                py, str(repo / "scripts" / "analyse_adaptation_session.py"),
                "--audit", str(audit_csv),
                "--session", str(session_csv),
                "--out", str(analysis_out),
            ],
            "Analyse adaptation session",
        )
        print(f"  Analysis output: {analysis_out}")
    else:
        print("\n  Skipping analysis — missing audit CSV or session CSV.")

    # 8c. Plot
    if audit_csv.exists():
        fig_dir = repo / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_out = fig_dir / f"adaptation_{pid.lower()}__fig01__mwl_timeline.png"
        plot_cmd = [
            py, str(repo / "scripts" / "plot_adaptation_session.py"),
            "--audit", str(audit_csv),
            "--out", str(plot_out),
        ]
        if session_csv and session_csv.exists():
            plot_cmd += ["--session", str(session_csv)]
        _run(plot_cmd, "Plot MWL timeline")
        print(f"  Figure saved: {plot_out}")
    else:
        print("\n  Skipping plot \u2014 audit CSV not found.")

    # 8d. Adaptation vs control performance comparison
    control_csv = ctx.get("control_session_csv")
    if session_csv and session_csv.exists() and control_csv and control_csv.exists():
        print("\n" + "-" * 60)
        print("  ADAPTATION vs CONTROL \u2014 session KPI summary")
        print("-" * 60)
        try:
            sys.path.insert(0, str(repo / "src"))
            from performance.summarise_openmatb_performance import (
                _collect_performance_rows,
                _compute_derived_kpis,
            )

            def _fmt(v) -> str:
                return f"{v:.4f}" if v is not None else "N/A"

            rows_adapt = _collect_performance_rows(session_csv)
            rows_ctrl  = _collect_performance_rows(control_csv)
            kpi_adapt  = _compute_derived_kpis(rows_adapt)
            kpi_ctrl   = _compute_derived_kpis(rows_ctrl)

            track_a  = kpi_adapt.get("tracking", {}).get("center_deviation_rmse")
            track_c  = kpi_ctrl.get("tracking",  {}).get("center_deviation_rmse")
            comms_a  = kpi_adapt.get("comms",    {}).get("accuracy")
            comms_c  = kpi_ctrl.get("comms",     {}).get("accuracy")
            sysmon_a = kpi_adapt.get("sysmon",   {}).get("accuracy")
            sysmon_c = kpi_ctrl.get("sysmon",    {}).get("accuracy")

            print(f"  {'KPI':<30s} {'ADAPTATION':>12s} {'CONTROL':>10s}")
            print(f"  {'-'*30} {'-'*12} {'-'*10}")
            print(f"  {'Tracking RMSE (lower=better)':<30s} {_fmt(track_a):>12s} {_fmt(track_c):>10s}")
            print(f"  {'Comms accuracy':<30s} {_fmt(comms_a):>12s} {_fmt(comms_c):>10s}")
            print(f"  {'Sysmon accuracy':<30s} {_fmt(sysmon_a):>12s} {_fmt(sysmon_c):>10s}")
            print()
            print(f"  adaptation={session_csv.name}")
            print(f"  control   ={control_csv.name}")
        except Exception as _cmp_exc:
            print(f"  WARNING: Could not compute condition comparison: {_cmp_exc}")
    else:
        _missing = []
        if not (session_csv and session_csv.exists()):
            _missing.append("adaptation CSV")
        if not (control_csv and control_csv.exists()):
            _missing.append("control CSV")
        print(f"\n  Skipping condition comparison \u2014 missing: {', '.join(_missing)}")


# ---------------------------------------------------------------------------
# Post-phase verification gates  (--post-phase-verify)
# ---------------------------------------------------------------------------

import collections as _collections

CheckResult = _collections.namedtuple("CheckResult", ["name", "critical", "passed", "detail"])


def _verify_phase_outputs(phase: int, ctx: dict, output_root: Path, pid: str, session: str) -> list:
    """Return a list of CheckResult for the given phase.

    All strings are ASCII-only to avoid UnicodeEncodeError on Windows cp1252
    terminals.  Callers should pass the result to _phase_gate().
    """
    results = []

    def ok(name: str, detail: str = "") -> None:
        results.append(CheckResult(name=name, critical=False, passed=True, detail=detail))

    def warn(name: str, detail: str) -> None:
        results.append(CheckResult(name=name, critical=False, passed=False, detail=detail))

    def fail(name: str, detail: str) -> None:
        results.append(CheckResult(name=name, critical=True, passed=False, detail=detail))

    def check_file(path: Path, label: str, min_kb: float = 0.0, critical: bool = True) -> bool:
        """Check a file exists and optionally has a minimum size. Returns True on pass."""
        if not path.exists():
            if critical:
                fail(label, f"File not found: {path}")
            else:
                warn(label, f"File not found: {path}")
            return False
        size_kb = path.stat().st_size / 1024
        if min_kb > 0 and size_kb < min_kb:
            if critical:
                fail(label, f"File too small: {size_kb:.1f} KB (expected >= {min_kb:.0f} KB): {path}")
            else:
                warn(label, f"File too small: {size_kb:.1f} KB (expected >= {min_kb:.0f} KB): {path}")
            return False
        ok(label, f"{size_kb:.1f} KB  ({path.name})")
        return True

    def check_xdf(path: Path, label_prefix: str, expected_duration_s: float = 0.0, tol_s: float = 30.0) -> None:
        """Deep XDF validation: load with pyxdf, check streams + optional duration + markers."""
        if not path.exists():
            fail(f"{label_prefix}: XDF exists", f"File not found: {path}")
            return
        size_kb = path.stat().st_size / 1024
        ok(f"{label_prefix}: XDF exists", f"{size_kb:.1f} KB")

        try:
            import pyxdf
        except ImportError:
            warn(f"{label_prefix}: pyxdf available", "pyxdf not installed -- deep XDF check skipped")
            return

        try:
            streams, _ = pyxdf.load_xdf(str(path))
        except Exception as exc:
            fail(f"{label_prefix}: XDF loads cleanly", f"pyxdf error: {exc}")
            return
        ok(f"{label_prefix}: XDF loads cleanly")

        n_streams = len(streams)
        if n_streams < 2:
            fail(f"{label_prefix}: >= 2 streams", f"Only {n_streams} stream(s) found")
        else:
            names = [s["info"]["name"][0] for s in streams]
            ok(f"{label_prefix}: >= 2 streams", f"{n_streams} streams: {', '.join(names)}")

        # Duration check (uses max timestamp across all streams)
        if expected_duration_s > 0:
            all_ts = []
            for s in streams:
                ts = s.get("time_stamps")
                if ts is not None and len(ts) > 0:
                    all_ts.extend([float(ts[0]), float(ts[-1])])
            if all_ts:
                span = max(all_ts) - min(all_ts)
                lo, hi = expected_duration_s - tol_s, expected_duration_s + tol_s
                detail = f"{span:.0f} s  (expected ~{expected_duration_s:.0f} s +/- {tol_s:.0f} s)"
                if lo <= span <= hi:
                    ok(f"{label_prefix}: duration", detail)
                else:
                    warn(f"{label_prefix}: duration", detail)

        # Marker check: look for any STUDY/V0/ marker in any string stream
        marker_found = False
        for s in streams:
            stype = s["info"]["type"][0] if s["info"]["type"] else ""
            if stype in ("Markers", "Events", ""):
                for sample in s.get("time_series", []):
                    if isinstance(sample, (list, tuple)):
                        val = sample[0] if sample else ""
                    else:
                        val = str(sample)
                    if "STUDY/V0/" in str(val):
                        marker_found = True
                        break
            if marker_found:
                break
        if marker_found:
            ok(f"{label_prefix}: STUDY/V0 markers present")
        else:
            warn(f"{label_prefix}: STUDY/V0 markers present", "No STUDY/V0/ marker found in XDF")

    # ------------------------------------------------------------------
    # Phase-specific checks
    # ------------------------------------------------------------------

    phys_dir = output_root / "physiology" / f"sub-{pid}" / f"ses-{session}" / "physio"
    pid_lower = pid.lower()

    if phase == 1:
        # Practice: just verify a session CSV exists somewhere (best-effort)
        sessions_dir = ctx["session_data_dir"] / "sessions"
        csvs = list(sessions_dir.glob("**/*.csv")) if sessions_dir.exists() else []
        if csvs:
            ok("Session CSV written", f"{len(csvs)} CSV file(s) in {sessions_dir.name}/")
        else:
            warn("Session CSV written", f"No CSV found under {sessions_dir}")

    elif phase == 2:
        # Staircase
        d_final = ctx.get("d_final")
        staircase_csv = ctx.get("staircase_csv")

        if staircase_csv and check_file(staircase_csv, "staircase_log.csv exists", min_kb=0.1):
            # Row count
            try:
                import csv as _csv
                with open(staircase_csv, newline="") as _f:
                    rows = list(_csv.DictReader(_f))
                n_rows = len(rows)
                if n_rows >= 5:
                    ok("Staircase ran >= 5 steps", f"{n_rows} rows")
                else:
                    warn("Staircase ran >= 5 steps", f"Only {n_rows} row(s) -- staircase may not have converged")
            except Exception as exc:
                warn("Staircase CSV readable", f"Could not parse: {exc}")

        if d_final is None:
            fail("d_final set", "d_final not found in context")
        elif not (0.05 <= d_final <= 0.95):
            fail("d_final in range [0.05, 0.95]", f"d_final = {d_final:.4f}")
        else:
            ok("d_final in range [0.05, 0.95]", f"d_final = {d_final:.4f}")
            if d_final <= 0.05 or d_final >= 0.95:
                warn("d_final not at ceiling/floor", f"d_final = {d_final:.4f} -- at extreme")

    elif phase == 3:
        # Rest baseline XDF
        candidates = sorted(phys_dir.glob("*acq-rest*.xdf"), key=lambda p: p.stat().st_mtime) if phys_dir.exists() else []
        if not candidates:
            fail("Rest baseline XDF found", f"No *acq-rest*.xdf in {phys_dir}")
        else:
            xdf = candidates[-1]
            check_xdf(xdf, "Rest XDF", expected_duration_s=120.0, tol_s=15.0)

    elif phase == 4:
        # Scenario files
        scenarios_dir = ctx["scenarios_dir"]
        for key, fname in [
            ("calibration_scenario_c1", ctx.get("calibration_scenario_c1", f"full_calibration_{pid_lower}_c1.txt")),
            ("calibration_scenario_c2", ctx.get("calibration_scenario_c2", f"full_calibration_{pid_lower}_c2.txt")),
            ("adaptation_scenario",     ctx.get("adaptation_scenario",     f"adaptive_automation_{pid_lower}_c1_8min.txt")),
        ]:
            p = scenarios_dir / fname
            if check_file(p, f"{fname} exists"):
                # Check header contains d_final
                d_final = ctx.get("d_final")
                if d_final is not None:
                    try:
                        header = p.read_text(encoding="utf-8", errors="replace")[:500]
                        d_str = f"{d_final:.4f}"
                        if d_str in header:
                            ok(f"{fname}: d_final in header", f"d_final = {d_str}")
                        else:
                            warn(f"{fname}: d_final in header", f"'{d_str}' not found in first 500 chars")
                    except Exception as exc:
                        warn(f"{fname}: d_final in header", f"could not read file: {exc}")

    elif phase == 5:
        # Calibration XDFs
        for c in (1, 2):
            candidates = sorted(phys_dir.glob(f"*acq-cal_c{c}*.xdf"), key=lambda p: p.stat().st_mtime) if phys_dir.exists() else []
            if not candidates:
                fail(f"Cal C{c} XDF found", f"No *acq-cal_c{c}*.xdf in {phys_dir}")
            else:
                xdf_path = candidates[-1]
                check_xdf(xdf_path, f"Cal C{c} XDF", expected_duration_s=540.0, tol_s=45.0)
                # Critical: verify Markers stream is present (required for correct block timing).
                # If absent, LabRecorder restarted before MATB launched and the fix in
                # run_openmatb.py did not fire correctly.
                try:
                    import pyxdf as _pyxdf
                    _streams, _ = _pyxdf.load_xdf(str(xdf_path))
                    _marker_streams = [s for s in _streams if s["info"]["type"][0] == "Markers"]
                    if _marker_streams:
                        _n_events = sum(len(s.get("time_series", [])) for s in _marker_streams)
                        ok(f"Cal C{c} XDF: Markers stream present", f"{len(_marker_streams)} stream(s), {_n_events} event(s)")
                    else:
                        fail(f"Cal C{c} XDF: Markers stream present",
                             "No Markers stream found — LabRecorder restart did not capture MATB outlet. "
                             "Block timing will fall back to hardcoded offset; calibration model will be unreliable.")
                except Exception as _exc:
                    warn(f"Cal C{c} XDF: Markers stream check", f"Could not inspect streams: {_exc}")

    elif phase == 6:
        # Model artefacts
        model_dir = ctx.get("participant_model_dir", output_root / "models" / pid)
        for fname in ("pipeline.pkl", "selector.pkl"):
            check_file(model_dir / fname, fname, min_kb=1.0)
        # norm_stats.json
        ns_path = model_dir / "norm_stats.json"
        if check_file(ns_path, "norm_stats.json"):
            try:
                ns = json.loads(ns_path.read_text(encoding="utf-8"))
                mean_len = len(ns.get("mean", []))
                std_len = len(ns.get("std", []))
                if mean_len == 54 and std_len == 54:
                    ok("norm_stats: mean/std length == 54", f"mean={mean_len}, std={std_len}")
                else:
                    fail("norm_stats: mean/std length == 54", f"mean={mean_len}, std={std_len} (expected 54)")
            except Exception as exc:
                fail("norm_stats.json parses", f"JSON error: {exc}")
        # model_config.json
        mc_path = model_dir / "model_config.json"
        if check_file(mc_path, "model_config.json"):
            try:
                mc = json.loads(mc_path.read_text(encoding="utf-8"))
                thresh = mc.get("youden_threshold")
                if thresh is None:
                    fail("model_config: youden_threshold present", "Key 'youden_threshold' missing")
                elif not (0.0 < thresh < 1.0):
                    fail("model_config: youden_threshold in (0, 1)", f"youden_threshold = {thresh}")
                else:
                    ok("model_config: youden_threshold in (0, 1)", f"youden_threshold = {thresh:.4f}")
                    # Informational
                    j_val = mc.get("youdens_j", mc.get("youden_j", "N/A"))
                    n_cls = mc.get("n_classes", "N/A")
                    ok("model_config info", f"Youden J = {j_val}, n_classes = {n_cls}")
            except Exception as exc:
                fail("model_config.json parses", f"JSON error: {exc}")

    elif phase == 70:
        # Pre-adaptation baseline refresh
        model_dir = ctx.get("participant_model_dir", output_root / "models" / pid)
        ns_path = model_dir / "norm_stats.json"
        if ns_path.exists():
            import time as _time
            age_s = _time.time() - ns_path.stat().st_mtime
            if age_s < 300:
                ok("norm_stats.json refreshed recently", f"Modified {age_s:.0f} s ago (< 300 s)")
            else:
                fail("norm_stats.json refreshed recently", f"Modified {age_s:.0f} s ago -- baseline refresh may not have run")
        else:
            fail("norm_stats.json exists after refresh", f"Not found: {ns_path}")

    elif phase == 71:
        # Adaptation condition
        audit_csv = ctx.get("audit_csv")
        if audit_csv and check_file(audit_csv, "mwl_audit.csv", min_kb=0.1):
            try:
                import csv as _csv
                with open(audit_csv, newline="") as _f:
                    rows = list(_csv.DictReader(_f))
                n_rows = len(rows)
                if n_rows >= 1:
                    ok("audit CSV has rows", f"{n_rows} rows")
                else:
                    warn("audit CSV has rows", "audit CSV is empty -- adaptation may not have been active")
                # Look for at least one assist_on toggle
                toggles = [r for r in rows if str(r.get("action", "")).strip() == "assist_on"]
                if toggles:
                    ok("At least 1 assist_on toggle fired", f"{len(toggles)} toggle(s)")
                else:
                    warn("At least 1 assist_on toggle fired", "No 'assist_on' rows found -- check MWL threshold / model quality")
            except Exception as exc:
                warn("audit CSV readable", f"Could not parse: {exc}")

    elif phase == 72:
        # Control condition
        sessions_dir = ctx["session_data_dir"] / "sessions"
        csvs = sorted(sessions_dir.glob("**/*.csv")) if sessions_dir.exists() else []
        if csvs:
            ok("Control session CSV written", f"{len(csvs)} CSV file(s)")
        else:
            fail("Control session CSV written", f"No CSVs found under {sessions_dir}")

    return results


def _phase_gate(phase: int, results: list, pid: str, original_argv: list, no_pause: bool = False) -> None:
    """Print a PASS/WARN/FAIL summary and either prompt to continue or exit on critical failure.

    All output is ASCII-only to avoid UnicodeEncodeError on Windows cp1252 terminals.
    On critical failure, prints the exact --start-phase restart command then exits.
    """
    phase_labels = {
        1: "Phase 1 (Practice)",
        2: "Phase 2 (Staircase)",
        3: "Phase 3 (Rest baseline)",
        4: "Phase 4 (Scenario generation)",
        5: "Phase 5 (Calibration runs)",
        6: "Phase 6 (Model calibration)",
        70: "Phase 7a (Pre-adaptation baseline refresh)",
        71: "Phase 7b (Adaptation condition)",
        72: "Phase 7c (Control condition)",
        8: "Phase 8 (Post-session)",
    }
    label = phase_labels.get(phase, f"Phase {phase}")

    divider = "-" * 50
    print(f"\n{divider}")
    print(f" {label} verification  (--post-phase-verify)")
    print(divider)

    n_crit = 0
    n_warn = 0
    for r in results:
        if r.passed:
            tag = "[PASS]"
        elif r.critical:
            tag = "[FAIL]"
            n_crit += 1
        else:
            tag = "[WARN]"
            n_warn += 1
        line = f"  {tag}  {r.name}"
        if r.detail:
            line += f"  -- {r.detail}"
        print(line)

    print()
    if n_crit == 0 and n_warn == 0:
        print(f"  All checks passed.")
    else:
        summary_parts = []
        if n_warn:
            summary_parts.append(f"{n_warn} warning(s)")
        if n_crit:
            summary_parts.append(f"{n_crit} CRITICAL failure(s)")
        print("  " + "  |  ".join(summary_parts))

    if n_crit > 0:
        # Build restart command: replace --start-phase or insert it
        restart_argv = [a for a in original_argv if not a.startswith("--start-phase")]
        # Remove the value after --start-phase if it was written as two tokens
        clean = []
        skip_next = False
        for tok in restart_argv:
            if skip_next:
                skip_next = False
                continue
            if tok == "--start-phase":
                skip_next = True
                continue
            clean.append(tok)
        # Map internal phase number back to user-facing phase (70/71/72 => 7)
        restart_phase = phase if phase <= 8 else 7
        restart_cmd = ["python"] + clean[1:] + ["--start-phase", str(restart_phase)]
        print(f"\n  To retry {label}:")
        print(f"    python {' '.join(restart_cmd[1:])}")
        print(divider)
        sys.exit(1)

    print(divider)
    if not no_pause:
        input(f"\n  Press ENTER to continue, or Ctrl-C to abort... ")
    else:
        print("  (--no-pause: auto-continuing)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end orchestrator for a single participant study session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--participant", required=True,
        help="Participant ID (e.g. P001).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=f"External data root (default: {_DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--group-model-dir", type=Path, required=False, default=None,
        help="Directory containing group_pipeline.pkl (from train-group). "
             "Not used by default — calibration now runs in scratch mode.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print session plan and exit without running anything.",
    )
    parser.add_argument(
        "--labrecorder-rcs", action="store_true",
        help="Pass --labrecorder-rcs to run_openmatb.py for XDF recording.",
    )
    parser.add_argument(
        "--verification", action="store_true",
        help="Pass --verification --skip-stream-check (for automated desk testing).",
    )
    parser.add_argument(
        "--speed", type=int, default=1,
        help="Fast-forward speed (only with --verification).",
    )
    parser.add_argument(
        "--eda-auto-port", action="store_true",
        help="Auto-detect the Shimmer COM port (passes --eda-auto-port to run_openmatb.py).",
    )
    parser.add_argument(
        "--skip-stream-check", action="store_true",
        help="Skip preflight LSL stream checks (passes --skip-stream-check to run_openmatb.py).",
    )
    parser.add_argument(
        "--start-phase", type=int, default=1, metavar="N",
        help="Skip all phases before N (1-8). Use to resume a session mid-way.",
    )
    parser.add_argument(
        "--end-phase", type=int, default=8, metavar="N",
        help="Stop after phase N and exit cleanly. Use 2 for staircase-only runs (no sensors needed).",
    )
    parser.add_argument(
        "--skip-baseline-refresh", action="store_true",
        help="Skip the pre-adaptation resting baseline refresh (phase 7.0). "
             "Use when the calibration rest recording is recent enough.",
    )
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip the dual-EEG smoke test before Phase 6 (useful when amps are already verified).",
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="Skip all interactive ENTER prompts (useful when resuming mid-session).",
    )
    parser.add_argument(
        "--post-phase-verify", action="store_true",
        help=(
            "After each phase, verify expected outputs and prompt before continuing. "
            "Prints [PASS]/[WARN]/[FAIL] for each check. "
            "Critical failures print the --start-phase N restart command and exit. "
            "Applies regardless of participant ID."
        ),
    )

    args = parser.parse_args()
    global _NO_PAUSE
    _NO_PAUSE = args.no_pause
    _check_venv()
    pid = args.participant
    output_root = args.output_root or _DEFAULT_OUTPUT_ROOT

    # --- Load config ---
    cfg = _load_participant_config(pid)
    adaptation_first = cfg.get("adaptation_first", True)
    skip_practice = cfg.get("skip_practice", False)
    session = _next_session_id(cfg)

    # --- Derive paths ---
    session_data_dir = output_root / "openmatb" / pid / session
    physiology_dir = output_root / "physiology" / pid / session
    model_out_dir = output_root / "models" / pid
    audit_csv = session_data_dir / "mwl_audit.csv"

    # --- Condition order ---
    if adaptation_first:
        condition_order = ["adaptation", "control"]
    else:
        condition_order = ["control", "adaptation"]

    # --- Session plan ---
    plan = [
        ("1", "Practice", "Familiarisation scenarios (no sensors required)" + (" [SKIPPED — skip_practice=true]" if skip_practice else "")),
        ("2", "Staircase", "Online staircase -> d_final (no sensors required)"),
        ("   --- SENSOR SETUP ---", "", "fit EEG, EDA, HR before Phase 3"),
        ("3", "Rest baseline", "2-min fixation cross (EEG normalisation)"),
        ("4", "Generate scenarios", "Calibration + adaptation from d_final"),
        ("5", "Calibration runs", "2 x 9-min counterbalanced (LabRecorder)"),
        ("6", "Model calibration", "Scratch 3-class SVM-linear (k=35, C=1.0) on participant's MATB cal data"),
        ("7", f"Condition A: {condition_order[0]}", "8-min block"),
        ("7", f"Condition B: {condition_order[1]}", "8-min block"),
        ("8", "Post-session", "Verification + analysis + plots"),
    ]

    print("\n" + "=" * 60)
    print(f"  FULL STUDY SESSION PLAN")
    print(f"  Participant : {pid}")
    print(f"  Session     : {session}")
    print(f"  Condition   : {'adaptation -> control' if adaptation_first else 'control -> adaptation'}")
    print(f"  Output root : {output_root}")
    print(f"  Model       : scratch 3-class (no group model required)")
    print("=" * 60)
    for num, name, desc in plan:
        if name == "":
            print(f"  {num}")
        else:
            print(f"  Phase {num}: {name:30s} -- {desc}")
    print("=" * 60)

    if args.dry_run:
        print("\n  --dry-run: exiting without running.\n")
        return 0

    start_phase = args.start_phase
    end_phase = args.end_phase
    if start_phase > 1:
        print(f"\n  Resuming from Phase {start_phase} (skipping 1-{start_phase - 1}).\n")
    if start_phase >= 3:
        print("  NOTE: Resuming at Phase >=3 -- all sensors must already be fitted.\n")

    # When skipping early phases, restore ctx keys that those phases would have set.
    if start_phase > 2:
        # Phase 2 (staircase) sets d_final/staircase_csv.
        # Identify the staircase run by finding the run manifest whose playlist
        # contains adaptation_skeleton.txt, then extract d_final from that CSV.
        _d_final = 0.8  # fallback if CSV not found or extraction fails
        _staircase_csv = None
        try:
            _run_manifests = sorted(
                (output_root / "openmatb" / pid / session).glob("run_manifest_*.json")
            )
            for _rm_path in _run_manifests:
                _rm = json.loads(_rm_path.read_text(encoding="utf-8"))
                _playlist = _rm.get("playlist", [])
                if any("adaptation_skeleton" in (e.get("scenario_filename") or "") for e in _playlist):
                    _csv_str = _playlist[0].get("session_csv")
                    if _csv_str and Path(_csv_str).exists():
                        _staircase_csv = Path(_csv_str)
                        break
            if _staircase_csv is None:
                raise FileNotFoundError("No run manifest with adaptation_skeleton.txt found")
            sys.path.insert(0, str(_REPO_ROOT / "scripts" / "generate_scenarios"))
            from generate_full_study_scenarios import extract_d_final as _extract_d_final
            _d_final = _extract_d_final(_staircase_csv)
            print(f"  [resume] d_final={_d_final:.4f}  (from {_staircase_csv.name})")
        except Exception as _exc:
            print(
                f"  WARNING: Could not read d_final from staircase CSV ({_exc}).\n"
                f"           Using fallback d_final={_d_final:.4f} — verify this is correct."
            )
        ctx_pre: dict = {"d_final": _d_final, "staircase_csv": _staircase_csv}
    else:
        ctx_pre = {}

    if start_phase > 3:
        # Phase 3 (rest baseline) sets rest_baseline_xdf for Phase 6 norm reference.
        _rest_xdf = (
            output_root / "physiology"
            / f"sub-{pid}" / f"ses-{session}" / "physio"
            / f"sub-{pid}_ses-{session}_task-matb_acq-rest_physio.xdf"
        )
        ctx_pre["rest_baseline_xdf"] = _rest_xdf if _rest_xdf.exists() else None

    if start_phase > 4:
        # Phase 4 sets scenario filenames
        _pid_lower = pid.lower()
        ctx_pre["calibration_scenario_c1"] = f"full_calibration_{_pid_lower}_c1.txt"
        ctx_pre["calibration_scenario_c2"] = f"full_calibration_{_pid_lower}_c2.txt"
        ctx_pre["adaptation_scenario"] = f"adaptive_automation_{_pid_lower}_c1_8min.txt"

    if start_phase > 6:
        # Phase 6 sets participant_model_dir and mwl_threshold.
        # Restore both so Phase 7 (adaptation) uses the correct Youden J cutoff.
        ctx_pre["participant_model_dir"] = output_root / "models" / pid
        _model_cfg_path = output_root / "models" / pid / "model_config.json"
        if _model_cfg_path.exists():
            _model_cfg = json.loads(_model_cfg_path.read_text(encoding="utf-8"))
            ctx_pre["mwl_threshold"] = _model_cfg["youden_threshold"]
            print(f"  [resume] mwl_threshold restored from model_config.json: {ctx_pre['mwl_threshold']:.4f}")
        else:
            print(
                f"  WARNING: model_config.json not found at {_model_cfg_path}\n"
                "           mwl_threshold will default to 0.5 — ensure Phase 6 ran successfully."
            )

    # --- Build context dict passed to all phases ---
    ctx = {
        "pid": pid,
        "session": session,
        "output_root": output_root,
        "session_data_dir": session_data_dir,
        "physiology_dir": physiology_dir,
        "model_out_dir": model_out_dir,
        "audit_csv": audit_csv,
        "group_model_dir": args.group_model_dir,
        "adaptation_first": adaptation_first,
        "condition_order": condition_order,
        "labrecorder_rcs": args.labrecorder_rcs,
        "eda_auto_port": args.eda_auto_port,
        "skip_stream_check": args.skip_stream_check,
        "skip_smoke_test": args.skip_smoke_test,
        "verification": args.verification,
        "speed": args.speed,
        "python": sys.executable,
        "repo_root": _REPO_ROOT,
        "scenarios_dir": _SCENARIOS_DIR,
        "post_phase_verify": args.post_phase_verify,
        "skip_practice": skip_practice,
        "skip_baseline_refresh": args.skip_baseline_refresh,
    }

    # Restore ctx keys that were set by any skipped phases.
    ctx.update(ctx_pre)

    # --- Execute phases ---
    _pause("Review session plan above. Ready to begin?")

    # Convenience: run post-phase gate when --post-phase-verify is active.
    def _gate(phase_num: int) -> None:
        if ctx["post_phase_verify"]:
            results = _verify_phase_outputs(phase_num, ctx, output_root, pid, session)
            _phase_gate(phase_num, results, pid, sys.argv[:], no_pause=args.no_pause)

    if start_phase <= 1 and not ctx["skip_practice"]:
        phase_practice(ctx)
        _gate(1)

    if start_phase <= 2 and end_phase >= 2:
        phase_staircase(ctx)
        _gate(2)

    if end_phase <= 2:
        print("\n" + "=" * 60)
        print(f"  STAIRCASE COMPLETE: {pid}")
        print(f"  d_final = {ctx.get('d_final', 'N/A')}")
        print("  Sensors can now be fitted. Run Full Session icon to continue.")
        print("=" * 60 + "\n")
        return 0

    if start_phase <= 2:
        _pause(
            "SENSOR PLACEMENT -- Fit all sensors before continuing:\n"
            "  1. EEG cap: apply gel, check impedances (target <25 kOhm),\n"
            "              confirm both amplifiers are powered and streaming.\n"
            "  2. Shimmer EDA: attach to non-dominant hand (index + middle finger),\n"
            "                  power on, confirm Bluetooth connected.\n"
            "  3. Polar HR: strap on chest, confirm BLE paired.\n"
            "  Press ENTER when all sensors are on, impedances accepted, and ready."
        )

    if start_phase <= 3:
        # LabRecorder is only needed from phase 3 onwards (EEG recording).
        if ctx["labrecorder_rcs"]:
            print("\n" + "=" * 60)
            print("  LABRECORDER")
            print("=" * 60)
            _ensure_labrecorder_running()
        phase_rest_baseline(ctx)
        _gate(3)
        _pause("Rest baseline complete. Ready to generate scenarios?")

    if start_phase <= 4:
        phase_generate_scenarios(ctx)
        _gate(4)

    if start_phase <= 5:
        if ctx["labrecorder_rcs"]:
            _ensure_labrecorder_running()
        phase_calibration_runs(ctx)
        _gate(5)
        _pause("Calibration runs complete. Ready for model calibration?")

    if start_phase <= 6:
        phase_model_calibration(ctx)
        _gate(6)

    # Ensure LabRecorder is running before the baseline and experimental
    # conditions start.  When resuming from --start-phase 7 the earlier phases
    # that would have launched it are skipped, so we guarantee it here.
    if ctx["labrecorder_rcs"]:
        _ensure_labrecorder_running()

    if ctx["skip_baseline_refresh"]:
        print("\n  [SKIPPED] Pre-adaptation baseline refresh (--skip-baseline-refresh)")
    else:
        phase_pre_adaptation_baseline(ctx)
        _gate(70)

    phase_experimental_conditions(ctx)
    _gate(71)
    _gate(72)
    _pause("Experimental conditions complete. Running post-session analysis...")

    phase_post_session(ctx)

    print("\n" + "=" * 60)
    print(f"  SESSION COMPLETE: {pid} / {session}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSession aborted by operator (Ctrl-C).")
        sys.exit(1)
