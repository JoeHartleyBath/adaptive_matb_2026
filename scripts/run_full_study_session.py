"""End-to-end orchestrator for a single participant study session.

Automates the full flow:
  1. Practice (4 familiarisation scenarios)
  2. Staircase calibration (adaptation_skeleton.txt)
  3. Scenario generation (calibration + adaptation scenarios from d_final)
  4. Calibration runs (2 × 9-min counterbalanced scenarios)
  5. Model calibration (warm-start LogReg from group model)
  6. Adaptation condition (generated scenario + MWL-driven toggle)
  7. Control condition (same scenario, no adaptation)
     — conditions 6 & 7 counterbalanced via adaptation_first in config
  8. Post-session verification + analysis + plots

Usage:
    python scripts/run_full_study_session.py --participant P001 \\
        --group-model-dir D:/models/group \\
        --output-root C:/data/adaptive_matb

The script pauses for operator confirmation between major phases.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import yaml
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASSIGNMENTS_PATH = _REPO_ROOT / "config" / "participant_assignments.yaml"
_SCENARIOS_DIR = _REPO_ROOT / "experiment" / "scenarios"

# Default output root matches run_openmatb.py convention.
_DEFAULT_OUTPUT_ROOT = Path(r"C:\data\adaptive_matb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _pause(msg: str) -> None:
    """Pause for operator confirmation. Ctrl-C aborts."""
    print(f"\n{'='*60}")
    print(f"  PAUSE: {msg}")
    print(f"{'='*60}")
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
    ]
    if ctx["verification"]:
        cmd += ["--verification", "--skip-stream-check", "--speed", str(ctx["speed"])]
    if ctx["labrecorder_rcs"]:
        cmd += ["--labrecorder-rcs"]
    return cmd


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


def phase_staircase(ctx: dict) -> None:
    """Phase 2: Staircase calibration → d_final."""
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

    _run(
        [
            py, str(ctx["repo_root"] / "scripts" / "calibrate_participant_logreg.py"),
            "calibrate",
            "--group-dir", str(ctx["group_model_dir"]),
            "--xdf-dir", str(xdf_dir),
            "--pid", pid,
            "--out-dir", str(model_out),
        ],
        "Calibrate participant model",
    )
    ctx["participant_model_dir"] = model_out
    print(f"\n  Model artefacts: {model_out}")


def phase_experimental_conditions(ctx: dict) -> None:
    """Phase 6+7: Adaptation and control conditions (counterbalanced)."""
    print("\n" + "=" * 60)
    print("  PHASE 6–7: EXPERIMENTAL CONDITIONS")
    print(f"  Order: {ctx['condition_order'][0]} → {ctx['condition_order'][1]}")
    print("=" * 60)
    scenario = ctx["adaptation_scenario"]

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

    args = parser.parse_args()
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
        ("1", "Practice", "4 familiarisation scenarios"),
        ("2", "Staircase", "Online staircase → d_final"),
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
    print(f"  Condition   : {'adaptation → control' if adaptation_first else 'control → adaptation'}")
    print(f"  Output root : {output_root}")
    print(f"  Group model : {args.group_model_dir}")
    print("=" * 60)
    for num, name, desc in plan:
        print(f"  Phase {num}: {name:30s} — {desc}")
    print("=" * 60)

    if args.dry_run:
        print("\n  --dry-run: exiting without running.\n")
        return 0

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
        "verification": args.verification,
        "speed": args.speed,
        "python": sys.executable,
        "repo_root": _REPO_ROOT,
        "scenarios_dir": _SCENARIOS_DIR,
    }

    # --- Execute phases ---
    _pause("Review session plan above. Ready to begin?")

    phase_practice(ctx)
    _pause("Practice complete. Ready for staircase?")

    phase_staircase(ctx)
    _pause("Staircase complete. Ready to generate scenarios?")

    phase_generate_scenarios(ctx)

    phase_calibration_runs(ctx)
    _pause("Calibration runs complete. Ready for model calibration?")

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
