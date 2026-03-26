"""End-to-end orchestrator for a single participant study session.

Automates the full flow:
  1. Rest baseline (2-min fixation cross for EEG normalisation)
  2. Staircase calibration (adaptation_skeleton.txt)
  3. Scenario generation (calibration + adaptation scenarios from d_final)
  4. Calibration runs (2 × 9-min counterbalanced scenarios)
  5. Model calibration (warm-start LogReg from group model)
  6. Adaptation condition (generated scenario + MWL-driven toggle)
  7. Control condition (same scenario, no adaptation)
     — conditions 6 & 7 counterbalanced via adaptation_first in config
  8. Post-session verification + analysis + plots

Note: practice scenarios are omitted — intended for researcher self-runs.

Usage:
    python scripts/run_full_study_session.py --participant P001 \\
        --group-model-dir D:/adaptive_matb_data/pretrain \\
        --output-root C:/adaptive_matb_2026/output

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
            f"Add them with scripts/generate_scenarios/generate_participant_assignments.py first."
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


def _run(cmd: list[str], label: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess with clear labelling. Exits on failure."""
    print(f"\n>>> [{label}]")
    print(f"    {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(f"\n!!! [{label}] failed with exit code {result.returncode}")
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
        "--seq-id", ctx["seq_id"],
        "--output-root", str(ctx["output_root"]),
        "--skip-assignment-update",
        "--pilot1",  # always: dual-amp EEG, physiology recording, XDF QC
    ]
    if ctx["verification"]:
        cmd += ["--verification", "--skip-stream-check", "--speed", str(ctx["speed"])]
    if ctx["labrecorder_rcs"]:
        cmd += ["--labrecorder-rcs"]
    if ctx.get("eda_auto_port"):
        cmd += ["--eda-auto-port"]
    return cmd


def _find_rest_xdf(ctx: dict) -> Path | None:
    """Return the resting-baseline XDF path written by LabRecorder, or None.

    LabRecorder saves to the BIDS physiology directory.  We search for the
    most-recently-modified ``*acq-rest*.xdf`` file there so we don't need to
    predict the exact BIDS filename.
    """
    phys_dir = (
        ctx["output_root"] / "physiology"
        / f"sub-{ctx['pid']}" / f"ses-{ctx['session']}" / "physio"
    )
    candidates = sorted(phys_dir.glob("*acq-rest*.xdf"), key=lambda p: p.stat().st_mtime)
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
    """Phase 1: Practice familiarisation (4 scenarios)."""
    print("\n" + "=" * 60)
    print("  PHASE 1: PRACTICE")
    print("=" * 60)
    for i, scenario in enumerate(_PRACTICE_SCENARIOS, 1):
        cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", scenario]
        _run(cmd, f"Practice {i}/{len(_PRACTICE_SCENARIOS)}: {scenario}")


def phase_rest_baseline(ctx: dict) -> None:
    """Phase 1: 2-min resting-state EEG baseline (fixation cross)."""
    print("\n" + "=" * 60)
    print("  PHASE 1: REST BASELINE")
    print("=" * 60)
    _pause(
        "EEG BASELINE — Instruct participant:\n"
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
    """Phase 2: Staircase calibration -> d_final."""
    print("\n" + "=" * 60)
    print("  PHASE 2: STAIRCASE CALIBRATION")
    print("=" * 60)
    cmd = _openmatb_base_cmd(ctx) + [
        "--only-scenario", "adaptation_skeleton.txt",
        "--adaptation",
    ]
    _run(cmd, "Staircase calibration")

    # Extract d_final from the staircase session CSV
    staircase_csv = _find_latest_session_csv(ctx)
    # Import the extraction function from the scenario generator
    sys.path.insert(0, str(ctx["repo_root"] / "scripts" / "generate_scenarios"))
    from generate_full_study_scenarios import extract_d_final
    d_final = extract_d_final(staircase_csv)
    ctx["d_final"] = d_final
    ctx["staircase_csv"] = staircase_csv
    print(f"\n  d_final = {d_final:.4f}  (from {staircase_csv.name})")


def phase_generate_scenarios(ctx: dict) -> None:
    """Phase 3: Generate calibration + adaptation scenarios from d_final."""
    print("\n" + "=" * 60)
    print("  PHASE 3: GENERATE SCENARIOS")
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
    """Phase 4: Two 9-min calibration runs (counterbalanced)."""
    print("\n" + "=" * 60)
    print("  PHASE 4: CALIBRATION RUNS")
    print("=" * 60)
    for condition in (1, 2):
        scenario = ctx[f"calibration_scenario_c{condition}"]
        cmd = _openmatb_base_cmd(ctx) + [
            "--only-scenario", scenario,
        ]
        if ctx["labrecorder_rcs"]:
            cmd += ["--labrecorder-acq", f"cal_c{condition}"]
        _run(cmd, f"Calibration run C{condition}: {scenario}")
        if condition == 1:
            _pause("Calibration run 1 complete. Collect TLX, then continue.")


def phase_model_calibration(ctx: dict) -> None:
    """Phase 5: Warm-start participant model from group model."""
    print("\n" + "=" * 60)
    print("  PHASE 5: MODEL CALIBRATION")
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
        py, str(ctx["repo_root"] / "scripts" / "calibrate_participant_logreg.py"),
        "calibrate",
        "--group-dir", str(ctx["group_model_dir"]),
        "--xdf-dir", str(xdf_dir),
        "--pid", pid,
        "--out-dir", str(model_out),
    ]
    rest_xdf = ctx.get("rest_baseline_xdf")
    if rest_xdf and rest_xdf.exists():
        calibrate_cmd += ["--resting-xdf", str(rest_xdf)]
        print(f"  Using resting XDF: {rest_xdf.name}")
    else:
        print("  No resting XDF — norm will use LOW block fallback.")

    _run(calibrate_cmd, "Calibrate participant model")
    ctx["participant_model_dir"] = model_out
    print(f"\n  Model artefacts: {model_out}")


def phase_experimental_conditions(ctx: dict) -> None:
    """Phase 6+7: Adaptation and control conditions (counterbalanced)."""
    print("\n" + "=" * 60)
    print("  PHASE 6–7: EXPERIMENTAL CONDITIONS")
    print(f"  Order: {ctx['condition_order'][0]} -> {ctx['condition_order'][1]}")
    print("=" * 60)
    scenario = ctx["adaptation_scenario"]

    # Smoke-test both EEG amps before any scenario launch (fast fail)
    if "adaptation" in ctx["condition_order"] and not ctx.get("skip_smoke_test"):
        if not _smoke_test_dual_eeg_inlet():
            sys.exit("Aborted: EEG dual-inlet smoke test failed. Fix EEG streams and re-run.")

    for i, condition in enumerate(ctx["condition_order"], 1):
        label = f"Condition {i}/{len(ctx['condition_order'])}: {condition}"
        cmd = _openmatb_base_cmd(ctx) + ["--only-scenario", scenario]

        if condition == "adaptation":
            cmd += [
                "--mwl-adaptation",
                "--mwl-model-dir", str(ctx["participant_model_dir"]),
                "--mwl-audit-csv", str(ctx["audit_csv"]),
            ]
        if ctx["labrecorder_rcs"]:
            cmd += ["--labrecorder-acq", condition]

        _run(cmd, label)

        # Capture the adaptation run's session CSV for post-session analysis
        if condition == "adaptation":
            ctx["adaptation_session_csv"] = _find_latest_session_csv(ctx)

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
        print("\n  Skipping plot — audit CSV not found.")


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
        "--group-model-dir", type=Path, required=True,
        help="Directory containing group_pipeline.pkl (from train-group).",
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
        "--start-phase", type=int, default=1, metavar="N",
        help="Skip all phases before N (1–8). Use to resume a session mid-way.",
    )
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip the dual-EEG smoke test before Phase 6 (useful when amps are already verified).",
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="Skip all interactive ENTER prompts (useful when resuming mid-session).",
    )

    args = parser.parse_args()
    global _NO_PAUSE
    _NO_PAUSE = args.no_pause
    _check_venv()
    pid = args.participant
    output_root = args.output_root or _DEFAULT_OUTPUT_ROOT

    # --- Load config ---
    cfg = _load_participant_config(pid)
    seq_id = cfg["sequence"]
    adaptation_first = cfg.get("adaptation_first", True)
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
        ("1", "Rest baseline", "2-min fixation cross (EEG normalisation)"),
        ("2", "Staircase", "Online staircase -> d_final"),
        ("3", "Generate scenarios", "Calibration + adaptation from d_final"),
        ("4", "Calibration runs", "2 × 9-min counterbalanced (LabRecorder)"),
        ("5", "Model calibration", "Warm-start LogReg from group model"),
        ("6", f"Condition A: {condition_order[0]}", "8-min experimental block"),
        ("7", f"Condition B: {condition_order[1]}", "8-min experimental block"),
        ("8", "Post-session", "Verification + analysis + plots"),
    ]

    print("\n" + "=" * 60)
    print(f"  FULL STUDY SESSION PLAN")
    print(f"  Participant : {pid}")
    print(f"  Session     : {session}")
    print(f"  Sequence    : {seq_id}")
    print(f"  Condition   : {'adaptation -> control' if adaptation_first else 'control -> adaptation'}")
    print(f"  Output root : {output_root}")
    print(f"  Group model : {args.group_model_dir}")
    print("=" * 60)
    for num, name, desc in plan:
        print(f"  Phase {num}: {name:30s} — {desc}")
    print("=" * 60)

    if args.dry_run:
        print("\n  --dry-run: exiting without running.\n")
        return 0

    start_phase = args.start_phase
    if start_phase > 1:
        print(f"\n  Resuming from Phase {start_phase} (skipping 1–{start_phase - 1}).\n")

    # When skipping early phases, restore ctx keys that those phases would have set.
    if start_phase > 1:
        # Phase 1 sets rest_baseline_xdf
        _rest_xdf = (
            output_root / "physiology"
            / f"sub-{pid}" / f"ses-{session}" / "physio"
            / f"sub-{pid}_ses-{session}_task-matb_acq-rest_physio.xdf"
        )
        ctx_pre: dict = {"rest_baseline_xdf": _rest_xdf if _rest_xdf.exists() else None}
    else:
        ctx_pre = {}

    if start_phase > 2:
        # Phase 2 sets d_final (needed by Phase 3).  Read from staircase CSV if present.
        import csv as _csv
        _staircase_csv = session_data_dir / "staircase_log.csv"
        _d_final = 0.8  # sensible default if CSV not found
        if _staircase_csv.exists():
            with open(_staircase_csv, newline="") as _f:
                _rows = list(_csv.DictReader(_f))
            if _rows:
                _d_final = float(_rows[-1].get("d", _d_final))
        ctx_pre["d_final"] = _d_final
        ctx_pre["staircase_csv"] = _staircase_csv

    if start_phase > 3:
        # Phase 3 sets scenario filenames
        _pid_lower = pid.lower()
        ctx_pre["calibration_scenario_c1"] = f"full_calibration_{_pid_lower}_c1.txt"
        ctx_pre["calibration_scenario_c2"] = f"full_calibration_{_pid_lower}_c2.txt"
        ctx_pre["adaptation_scenario"] = f"adaptive_automation_{_pid_lower}_c1_8min.txt"

    if start_phase > 5:
        # Phase 5 sets participant_model_dir
        ctx_pre["participant_model_dir"] = output_root / "models" / pid

    # --- Build context dict passed to all phases ---
    ctx = {
        "pid": pid,
        "session": session,
        "seq_id": seq_id,
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
        "skip_smoke_test": args.skip_smoke_test,
        "verification": args.verification,
        "speed": args.speed,
        "python": sys.executable,
        "repo_root": _REPO_ROOT,
        "scenarios_dir": _SCENARIOS_DIR,
    }

    # Restore ctx keys that were set by any skipped phases.
    ctx.update(ctx_pre)

    # --- Ensure LabRecorder is running before any phase needs it ---
    if ctx["labrecorder_rcs"]:
        print("\n" + "=" * 60)
        print("  LABRECORDER")
        print("=" * 60)
        _ensure_labrecorder_running()

    # --- Execute phases ---
    _pause("Review session plan above. Ready to begin?")

    if start_phase <= 1:
        phase_rest_baseline(ctx)
        _pause("Rest baseline complete. Ready for staircase?")

    if start_phase <= 2:
        phase_staircase(ctx)
        _pause("Staircase complete. Ready to generate scenarios?")

    if start_phase <= 3:
        phase_generate_scenarios(ctx)

    if start_phase <= 4:
        phase_calibration_runs(ctx)
        _pause("Calibration runs complete. Ready for model calibration?")

    if start_phase <= 5:
        phase_model_calibration(ctx)

    phase_experimental_conditions(ctx)
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
