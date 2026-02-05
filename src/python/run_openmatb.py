r"""Launch OpenMATB with repo-safe output paths.

This wrapper enforces that participant/session identifiers are provided and configures
OpenMATB to write session logs outside the git repo.

Usage (PowerShell):
  cd src/python/vendor/openmatb
        .\.venv\Scripts\Activate.ps1
    python ..\..\run_openmatb.py --participant P001 --session S001 --seq-id SEQ1

Environment variables (optional):
  OPENMATB_OUTPUT_ROOT   (default: C:\data\adaptive_matb)
  OPENMATB_PARTICIPANT / OPENMATB_PARTICIPANT_ID
  OPENMATB_SESSION / OPENMATB_SESSION_ID
    OPENMATB_SEQ_ID

OpenMATB uses:
  OPENMATB_OUTPUT_ROOT and OPENMATB_OUTPUT_SUBDIR
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import shutil
import sys
import yaml
from pathlib import Path
from typing import Optional


def _get_env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _validate_id(value: str, *, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} is required")

    # Keep folder names safe and predictable
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"{label} must be alphanumeric/underscore/dash (got: {value!r})")

    return value


def _load_assignments(repo_root: Path) -> dict:
    """Load participant assignments from config file."""
    assignments_path = repo_root / "config" / "participant_assignments.yaml"
    if not assignments_path.exists():
        return {"participants": {}}
    
    with open(assignments_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    return data if "participants" in data else {"participants": data}


def _save_assignments(repo_root: Path, assignments: dict, dry_run: bool = False) -> None:
    """Save participant assignments to config file."""
    if dry_run:
        return
    
    assignments_path = repo_root / "config" / "participant_assignments.yaml"
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(assignments_path, 'w', encoding='utf-8') as f:
        yaml.dump(assignments, f, default_flow_style=False, sort_keys=True)


def _get_recent_participants(assignments: dict, limit: int = 5) -> list[str]:
    """Get list of recent participants sorted by last_run timestamp."""
    participants = assignments.get("participants", {})
    
    # Filter participants with last_run timestamp
    recent = []
    for pid, data in participants.items():
        if data.get("last_run"):
            recent.append((pid, data["last_run"]))
    
    # Sort by timestamp descending
    recent.sort(key=lambda x: x[1], reverse=True)
    
    return [pid for pid, _ in recent[:limit]]


def _list_manifest_paths(sessions_dir: Path) -> set[Path]:
    if not sessions_dir.exists():
        return set()
    return {p.resolve() for p in sessions_dir.glob("**/*.manifest.json") if p.is_file()}


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_seq_id_into_manifest(
    manifest_path: Path,
    *,
    seq_id: str,
    dry_run: bool,
    scenario_filename: str,
    abort_reason: Optional[str] = None,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["seq_id"] = seq_id
    manifest["unattended"] = False
    manifest["dry_run"] = dry_run
    if abort_reason:
        manifest["abort_reason"] = abort_reason
    if scenario_filename:
        manifest["scenario_name"] = Path(scenario_filename).stem
        openmatb_meta = manifest.get("openmatb")
        if not isinstance(openmatb_meta, dict):
            openmatb_meta = {}
            manifest["openmatb"] = openmatb_meta
        openmatb_meta["scenario_path"] = scenario_filename
    identifiers = manifest.get("identifiers")
    if not isinstance(identifiers, dict):
        identifiers = {}
        manifest["identifiers"] = identifiers
    identifiers["seq_id"] = seq_id

    _atomic_write_json(manifest_path, manifest)


def _get_playlist(seq_id: str, dry_run: bool, *, calibration_only: bool = False) -> list[str]:
    if dry_run:
        return ["pilot_dry_run_v0.txt"]

    playlist: list[str] = []
    if not calibration_only:
        # Fixed training sequence
        playlist.extend(
            [
                "pilot_practice_intro.txt",
                "pilot_practice_low.txt",
                "pilot_practice_moderate.txt",
                "pilot_practice_high.txt",
            ]
        )

    # calibration blocks based on counterbalancing sequence
    # SEQ1: Low -> Moderate -> High
    # SEQ2: Moderate -> High -> Low
    # SEQ3: High -> Low -> Moderate
    calibration_levels = {
        "SEQ1": ["LOW", "MODERATE", "HIGH"],
        "SEQ2": ["MODERATE", "HIGH", "LOW"],
        "SEQ3": ["HIGH", "LOW", "MODERATE"],
    }

    try:
        levels = calibration_levels[seq_id]
    except KeyError as exc:
        raise ValueError(f"Unknown sequence ID: {seq_id}") from exc

    for level in levels:
        playlist.append(f"pilot_calibration_{level.lower()}.txt")

    return playlist


def _stage_pilot_instruction_files(openmatb_dir: Path, repo_root: Path) -> None:
    """Copy repo-managed pilot instruction text files into OpenMATB includes.

    OpenMATB validates blocking-plugin filenames by requiring them to exist under
    includes/instructions/ or includes/questionnaires/.
    """

    source_dir = repo_root / "instructions"
    required_names = [
        "1_welcome.txt",
        "2_sysmon.txt",
        "3_track.txt",
        "4_comm.txt",
        "5_resman.txt",
        "6_all_tasks.txt",
    ]

    missing = [name for name in required_names if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pilot instruction files under <repo>/instructions: " + ", ".join(missing)
        )

    # Keep pilot assets namespaced to avoid collisions with vendor examples.
    target_dir = openmatb_dir / "includes" / "instructions" / "pilot_en"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required_names:
        shutil.copyfile(source_dir / name, target_dir / name)


def _rewrite_scenario_paths_for_openmatb_includes(scenario_text: str) -> str:
    """Rewrite repo scenario text for OpenMATB runtime/validation compatibility."""

    # The vendor validator requires filenames for blocking plugins to be present under
    # includes/instructions/ or includes/questionnaires/.
    # Our repo scenarios may reference historical paths like ../../../../assets/…
    # Rewrite those to the namespaced include folder we populate.
    rewrites = [
        ("../../../../assets/instructions/pilot_en/", "pilot_en/"),
        ("..\\..\\..\\..\\assets\\instructions\\pilot_en\\", "pilot_en/"),
        ("..\\..\\..\\..\\assets\\instructions\\pilot_en/", "pilot_en/"),
        ("../../../../assets/instructions/pilot_en\\", "pilot_en/"),
    ]
    for old, new in rewrites:
        scenario_text = scenario_text.replace(old, new)

    # Normalize vendor parameter keys that are case-sensitive.
    # The ResMan plugin uses lowercase tank letters in parameter names
    # (e.g., tank-a-lossperminute). Some generated scenarios used tank-A-…
    # which fails OpenMATB validation.
    scenario_text = re.sub(
        r"\btank-([A-F])-",
        lambda m: f"tank-{m.group(1).lower()}-",
        scenario_text,
    )

    # Vendor genericscales plugin expects:
    #   <t>;genericscales;filename;<questionnaire.txt>
    #   <t>;genericscales;start
    # Some generated scenarios used genericscales;create and genericscales;load, which are invalid.
    scenario_text = re.sub(
        r"^\s*\d+:\d+:\d+;genericscales;create\s*$\r?\n?",
        "",
        scenario_text,
        flags=re.MULTILINE,
    )
    scenario_text = scenario_text.replace(";genericscales;load;", ";genericscales;filename;")
    return scenario_text


def _run_single_scenario(
    openmatb_dir: Path,
    scenario_filename: str,
    output_root_path: Path,
    participant: str,
    session: str,
    seq_id: str,
    args: argparse.Namespace,
    repo_commit: str,
    submodule_commit: str,
) -> tuple[int, Optional[Path]]:
    repo_root = Path(__file__).resolve().parents[2]

    try:
        _stage_pilot_instruction_files(openmatb_dir, repo_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2, None

    scenario_source_path = repo_root / "scenarios" / scenario_filename
    if not scenario_source_path.exists():
        print(f"Scenario file not found: {scenario_source_path}", file=sys.stderr)
        return 2, None

    scenario_target_path = openmatb_dir / "includes" / "scenarios" / scenario_filename
    scenario_target_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_text = scenario_source_path.read_text(encoding="utf-8")
    scenario_text = _rewrite_scenario_paths_for_openmatb_includes(scenario_text)
    scenario_target_path.write_text(scenario_text, encoding="utf-8")

    print(f"\n>>> Starting Scenario Block: {scenario_filename}")

    # Reset/Update environment for this specific run
    env = os.environ.copy()
    env["OPENMATB_OUTPUT_ROOT"] = str(output_root_path)
    env["OPENMATB_OUTPUT_SUBDIR"] = str(Path("openmatb") / participant / session)
    env["OPENMATB_PARTICIPANT"] = participant
    env["OPENMATB_SESSION"] = session
    env["OPENMATB_SEQ_ID"] = seq_id
    env["OPENMATB_REPO_COMMIT"] = repo_commit
    env["OPENMATB_SUBMODULE_COMMIT"] = submodule_commit

    # Calculate paths specifically for this run
    scenario_rel_path = scenario_filename
    sessions_dir = output_root_path / Path(env["OPENMATB_OUTPUT_SUBDIR"]) / "sessions"
    manifests_before = _list_manifest_paths(sessions_dir)

    # Generate bootstrap script dynamically for the scenario
    bootstrap = (
        "import sys\n"
        f"sys.argv.extend(['--speed', '{args.speed}'])\n"
        "import gettext, os, runpy\n"
        "from pathlib import Path\n"
        "\n"
        "# Wire --speed into the OpenMATB scenario clock (vendor does not parse argv for this).\n"
        "_speed = 1\n"
        "try:\n"
        "    if '--speed' in sys.argv:\n"
        "        i = sys.argv.index('--speed')\n"
        "        _speed = int(float(sys.argv[i + 1]))\n"
        "except Exception:\n"
        "    _speed = 1\n"
        "if _speed < 1:\n"
        "    _speed = 1\n"
        "try:\n"
        "    from core.clock import Clock\n"
        "    Clock._speed = _speed\n"
        "except Exception:\n"
        "    pass\n"
        "LOCALE_PATH = Path('.', 'locales')\n"
        "language_iso = [l for l in open('config.ini', 'r').readlines() if 'language=' in l][0].split('=')[-1].strip()\n"
        "language = gettext.translation('openmatb', LOCALE_PATH, [language_iso])\n"
        "language.install()\n"
        "from core.constants import CONFIG\n"
        "if not CONFIG.has_section('Openmatb'):\n"
        "    CONFIG.add_section('Openmatb')\n"
        f"CONFIG.set('Openmatb', 'scenario_path', {scenario_rel_path!r})\n"
        "# Some vendor plugins call self.log_manual_entry(), but AbstractPlugin only exposes self.logger.\n"
        "# Provide a compatibility shim to avoid runtime AttributeError.\n"
        "try:\n"
        "    from plugins.abstractplugin import AbstractPlugin\n"
        "    def _log_manual_entry(self, entry, key='manual'):\n"
        "        try:\n"
        "            self.logger.log_manual_entry(entry, key=key)\n"
        "        except Exception:\n"
        "            pass\n"
        "    AbstractPlugin.log_manual_entry = _log_manual_entry\n"
        "except Exception:\n"
        "    pass\n"
        "import core.event as event_mod\n"
        "_orig_parse = event_mod.Event.parse_from_string\n"
        "def _replace_tokens(text):\n"
        "    token_map = {\n"
        "        '${OPENMATB_PARTICIPANT}': os.environ.get('OPENMATB_PARTICIPANT', ''),\n"
        "        '${OPENMATB_SESSION}': os.environ.get('OPENMATB_SESSION', ''),\n"
        "        '${OPENMATB_SEQ_ID}': os.environ.get('OPENMATB_SEQ_ID', ''),\n"
        "    }\n"
        "    for key, value in token_map.items():\n"
        "        text = text.replace(key, value)\n"
        "    return text\n"
        "def _patched_parse(cls, line_id, line_str):\n"
        "    return _orig_parse(line_id, _replace_tokens(line_str))\n"
        "event_mod.Event.parse_from_string = classmethod(_patched_parse)\n"
        "from core.window import Window\n"
        "from core.modaldialog import ModalDialog\n"
        "from core.utils import get_conf_value\n"
        "from core.constants import REPLAY_MODE\n"
        "def _display_session_id(self):\n"
        "    if not REPLAY_MODE and get_conf_value('Openmatb', 'display_session_number'):\n"
        "        pid = os.environ.get('OPENMATB_PARTICIPANT') or 'UNKNOWN'\n"
        "        sid = os.environ.get('OPENMATB_SESSION') or 'UNKNOWN'\n"
        "        seq = os.environ.get('OPENMATB_SEQ_ID') or 'UNKNOWN'\n"
        "        msg = [f'Participant ID: {pid}', f'Session ID: {sid}', f'Sequence ID: {seq}']\n"
        "        self.modal_dialog = ModalDialog(self, msg, 'OpenMATB')\n"
        "Window.display_session_id = _display_session_id\n"
        "runpy.run_path('main.py', run_name='__main__')\n"
    )

    # Pass the modified environment
    proc = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=str(openmatb_dir), env=env)
    exit_code = proc.wait()

    # Post-process manifests
    if exit_code != 0:
        manifests_after = _list_manifest_paths(sessions_dir)
        new_manifests = sorted(manifests_after - manifests_before)
        if len(new_manifests) == 1:
            _write_seq_id_into_manifest(
                new_manifests[0],
                seq_id=seq_id,
                dry_run=args.dry_run,
                scenario_filename=scenario_filename,
                abort_reason=f"exit_code_{exit_code}",
            )
        return exit_code, (new_manifests[0] if len(new_manifests) == 1 else None)

    manifests_after = _list_manifest_paths(sessions_dir)
    new_manifests = sorted(manifests_after - manifests_before)
    if len(new_manifests) != 1:
        print(
            f"WARNING: Expected 1 new manifest for {scenario_filename}, found {len(new_manifests)} in {sessions_dir}. "
            "Skipping manifest metadata injection.",
            file=sys.stderr,
        )
        return 2, None

    manifest_path = new_manifests[0]
    _write_seq_id_into_manifest(
        manifest_path,
        seq_id=seq_id,
        dry_run=args.dry_run,
        scenario_filename=scenario_filename,
        abort_reason=None,
    )

    if getattr(args, "summarise_performance", False):
        try:
            summariser = (repo_root / "src" / "python" / "summarise_openmatb_performance.py").resolve()
            subprocess.run(
                [sys.executable, str(summariser), "--manifest", str(manifest_path)],
                check=False,
                cwd=str(repo_root),
            )
        except Exception:
            # Performance summarization is best-effort and must not crash an attended run.
            pass

    # If OpenMATB reports scenario parsing/runtime errors, treat this run as failed.
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        errors_log = Path(manifest.get("paths", {}).get("scenario_errors_log", ""))

        def _errors_log_has_actionable_errors(path: Path) -> bool:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return False
            stripped = text.strip()
            if not stripped:
                return False
            # OpenMATB writes a sentinel line when there are no validation errors.
            if stripped.lower() in {"no error", "no errors"}:
                return False
            return True

        if errors_log and errors_log.exists() and _errors_log_has_actionable_errors(errors_log):
            per_run_errors_log = manifest_path.with_suffix(manifest_path.suffix + ".errors.log")
            try:
                shutil.copyfile(errors_log, per_run_errors_log)
                manifest.setdefault("paths", {})["scenario_errors_log"] = str(per_run_errors_log)
                _atomic_write_json(manifest_path, manifest)
            except Exception:
                per_run_errors_log = errors_log
            _write_seq_id_into_manifest(
                manifest_path,
                seq_id=seq_id,
                dry_run=args.dry_run,
                scenario_filename=scenario_filename,
                abort_reason="scenario_errors",
            )
            print(f"Scenario errors detected; see: {per_run_errors_log}", file=sys.stderr)
            return 2, manifest_path

        # Detect early termination even without explicit errors: if the CSV ends far before
        # the latest timestamp in the scenario file, treat it as an abort.
        csv_path_str = manifest.get("paths", {}).get("session_csv")
        if csv_path_str:
            csv_path = Path(csv_path_str)
        else:
            csv_path = None

        def _scenario_max_time_seconds(path: Path) -> int:
            max_sec = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split(";")
                if not parts:
                    continue
                time_str = parts[0]
                try:
                    h, m, s = time_str.split(":")
                    sec = int(h) * 3600 + int(m) * 60 + int(s)
                except Exception:
                    continue
                if sec > max_sec:
                    max_sec = sec
            return max_sec

        def _csv_max_scenario_time_seconds(path: Path) -> float:
            try:
                with open(path, "r", encoding="utf-8", newline="") as f_csv:
                    header = f_csv.readline()
                    if not header:
                        return 0.0
                    cols = [c.strip() for c in header.split(",")]
                    try:
                        idx = cols.index("scenario_time")
                    except ValueError:
                        return 0.0
                    max_t = 0.0
                    for row in f_csv:
                        parts = row.rstrip("\n").split(",")
                        if len(parts) <= idx:
                            continue
                        try:
                            t = float(parts[idx])
                        except Exception:
                            continue
                        if t > max_t:
                            max_t = t
                    return max_t
            except Exception:
                return 0.0

        if csv_path and csv_path.exists():
            expected_end = _scenario_max_time_seconds(scenario_source_path)
            observed_end = _csv_max_scenario_time_seconds(csv_path)
            # Allow some slack (UI pauses, logging granularity), but if we didn't even
            # reach ~90% of intended time, assume the run ended early.
            if expected_end >= 10 and observed_end < (expected_end * 0.9):
                _write_seq_id_into_manifest(
                    manifest_path,
                    seq_id=seq_id,
                    dry_run=args.dry_run,
                    scenario_filename=scenario_filename,
                    abort_reason=f"early_exit_observed_{observed_end:.3f}_expected_{expected_end}",
                )
                print(
                    f"Scenario ended early (observed scenario_time={observed_end:.2f}s, expected~{expected_end}s)",
                    file=sys.stderr,
                )
                return 2, manifest_path
    except Exception:
        # If we can't read the manifest or error log, do not crash the runner.
        pass
    return 0, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenMATB with external output paths.")
    parser.add_argument(
        "--participant",
        help="Participant ID (e.g., P001). Can also be set via OPENMATB_PARTICIPANT.",
    )
    parser.add_argument(
        "--session",
        help="Session ID (e.g., S001). Can also be set via OPENMATB_SESSION.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="External data root (default: C:\\data\\adaptive_matb or OPENMATB_OUTPUT_ROOT).",
    )
    parser.add_argument(
        "--openmatb-dir",
        default=None,
        help="Path to OpenMATB directory (default: <repo>/src/python/vendor/openmatb).",
    )

    parser.add_argument(
        "--seq-id",
        required=False,
        choices=("SEQ1", "SEQ2", "SEQ3"),
        help="calibration-order sequence ID (SEQ1/SEQ2/SEQ3). Can also be set via OPENMATB_SEQ_ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic dry-run scenario artifact.",
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Enable fast-forward and other automation-oriented behaviors (verification only).",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Fast-forward speed multiplier (ignored unless --verification is set).",
    )

    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run only the counterbalanced calibration blocks (skip practice/training).",
    )

    parser.add_argument(
        "--pilot1",
        action="store_true",
        help=(
            "Enable Pilot 1 checks: requires a calibration-stage .xdf (LabRecorder) so a run-level manifest "
            "can link physiology to OpenMATB outputs and run alignment QC."
        ),
    )

    parser.add_argument(
        "--xdf-path",
        default=None,
        help=(
            "Path to the LabRecorder .xdf file that contains OpenMATB markers + physiology for this run. "
            "Required for --pilot1."
        ),
    )

    parser.add_argument(
        "--skip-xdf-qc",
        action="store_true",
        help="Skip XDF↔CSV marker alignment QC (pilot verification).",
    )

    parser.add_argument(
        "--summarise-performance",
        action="store_true",
        help="Write a derived performance summary JSON next to each run manifest.",
    )
    parser.add_argument(
        "--skip-assignment-update",
        action="store_true",
        help="Don't update participant_assignments.yaml (dry-run for assignments).",
    )

    args = parser.parse_args()

    if args.pilot1 and args.dry_run:
        print("NOTE: --pilot1 is ignored during --dry-run.", file=sys.stderr)
        args.pilot1 = False

    if args.speed != 1 and not args.verification:
        print(
            f"NOTE: Ignoring --speed={args.speed} because this is an attended run. "
            "Use --verification to enable fast-forward.",
            file=sys.stderr,
        )
        args.speed = 1

    repo_root = Path(__file__).resolve().parents[2]
    
    # Load participant assignments
    assignments = _load_assignments(repo_root)
    participants_data = assignments.get("participants", {})

    # Interactive prompts when arguments not provided
    participant_raw = args.participant or _get_env_first("OPENMATB_PARTICIPANT", "OPENMATB_PARTICIPANT_ID")
    session_raw = args.session or _get_env_first("OPENMATB_SESSION", "OPENMATB_SESSION_ID")
    seq_id = args.seq_id or _get_env_first("OPENMATB_SEQ_ID")

    # Interactive participant selection
    if participant_raw is None:
        print("\n=== OpenMATB Session Setup ===")
        
        # Show recent participants
        recent = _get_recent_participants(assignments, limit=5)
        if recent:
            print("\nRecent participants:")
            for i, pid in enumerate(recent, 1):
                pdata = participants_data[pid]
                seq = pdata.get("sequence", "?")
                completed = len(pdata.get("sessions_completed", []))
                print(f"  {i}. {pid} ({seq}, {completed} sessions)")
            print(f"  Or enter a participant number/ID")
        
        while not participant_raw:
            user_input = input("\nEnter participant number: ").strip()
            if not user_input:
                print("Participant selection cannot be empty.")
                continue
            
            # Check if selecting from recent list
            if user_input.isdigit() and 1 <= int(user_input) <= len(recent):
                participant_raw = recent[int(user_input) - 1]
            else:
                # Parse as participant number
                if not user_input.startswith('P'):
                    try:
                        num = int(user_input)
                        participant_raw = f"P{num:03d}"
                    except ValueError:
                        participant_raw = f"P{user_input}"
                else:
                    participant_raw = user_input
    
    # Look up sequence from assignments
    if participant_raw in participants_data:
        assigned_seq = participants_data[participant_raw].get("sequence")
        assigned_sessions = participants_data[participant_raw].get("sessions_completed", [])
        
        # Use assigned sequence if not overridden
        if seq_id is None:
            seq_id = assigned_seq
        elif seq_id != assigned_seq and assigned_seq:
            print(f"\nWARNING: Overriding assigned sequence {assigned_seq} with {seq_id}", file=sys.stderr)
        
        # Auto-increment session if needed
        if session_raw is None:
            next_session_num = len(assigned_sessions) + 1
            session_raw = f"S{next_session_num:03d}"
    else:
        # New participant not in assignments
        print(f"\nERROR: {participant_raw} not found in assignments file.", file=sys.stderr)
        print(f"Add to config/participant_assignments.yaml first, or use:", file=sys.stderr)
        print(f"  python scripts/generate_participant_assignments.py --participant-ids {participant_raw} --sequences SEQ1", file=sys.stderr)
        return 2
    
    # Confirmation prompt
    if not args.dry_run:
        print(f"\n{'='*50}")
        print(f"  Participant: {participant_raw}")
        print(f"  Sequence:    {seq_id}")
        print(f"  Session:     {session_raw}")
        print(f"{'='*50}")
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm not in ('y', 'yes'):
            print("Aborted by user.")
            return 1

    try:
        participant = _validate_id(participant_raw, label="participant")
        session = _validate_id(session_raw, label="session")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_root = args.output_root or _get_env_first("OPENMATB_OUTPUT_ROOT") or r"C:\data\adaptive_matb"
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        print(f"OPENMATB_OUTPUT_ROOT must be an absolute path (got: {output_root})", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    openmatb_dir = Path(args.openmatb_dir) if args.openmatb_dir else repo_root / "src" / "python" / "vendor" / "openmatb"

    if not openmatb_dir.exists():
        print(f"OpenMATB directory not found: {openmatb_dir}", file=sys.stderr)
        return 2

    def _git_rev_parse_head(cwd: Path) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.STDOUT)
        except Exception as exc:
            raise RuntimeError(f"Unable to determine git commit hash in {cwd}: {exc}") from exc
        return out.decode("utf-8", errors="replace").strip()

    try:
        repo_commit = _git_rev_parse_head(repo_root)
        submodule_commit = _git_rev_parse_head(openmatb_dir)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        playlist = _get_playlist(seq_id, args.dry_run, calibration_only=bool(args.calibration_only))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    missing_scenarios: list[str] = []
    for scenario_filename in playlist:
        if not (repo_root / "scenarios" / scenario_filename).exists():
            missing_scenarios.append(scenario_filename)
    if missing_scenarios:
        print("Missing scenario files under <repo>/scenarios:", file=sys.stderr)
        for name in missing_scenarios:
            print(f" - {name}", file=sys.stderr)
        return 2

    print(f"Running sequence: {seq_id}")
    print(f"Playlist ({len(playlist)} scenarios):")
    for s in playlist:
        print(f" - {s}")

    scenario_manifests: list[Path] = []

    for scenario_filename in playlist:
        exit_code, manifest_path = _run_single_scenario(
            openmatb_dir=openmatb_dir,
            scenario_filename=scenario_filename,
            output_root_path=output_root_path,
            participant=participant,
            session=session,
            seq_id=seq_id,
            args=args,
            repo_commit=repo_commit,
            submodule_commit=submodule_commit,
        )

        if exit_code != 0:
            print(f"\n!!! Scenario {scenario_filename} failed (code {exit_code}). Stopping sequence. !!!", file=sys.stderr)
            return exit_code

        if manifest_path:
            scenario_manifests.append(manifest_path)
        
        # Simple separation between blocks
        print(f"Scenario {scenario_filename} completed successfully.")
        # (Interactive UI and blocking dialogs are expected in attended mode)

    print("\nAll scenarios in playlist completed successfully.")

    if args.pilot1 and not args.xdf_path:
        # We require an explicit .xdf to create the run-level manifest and enforce QC.
        xdf_in = input("\nEnter LabRecorder .xdf path for calibration-stage physiology (e.g., *.xdf): ").strip()
        if not xdf_in:
            print("ERROR: --pilot1 requires an .xdf path (LabRecorder output).", file=sys.stderr)
            return 2
        args.xdf_path = xdf_in

    # Write run-level manifest linking the playlist to OpenMATB outputs (and optional physiology XDF).
    try:
        session_root = output_root_path / "openmatb" / participant / session
        run_manifest_dir = session_root
        run_manifest_dir.mkdir(parents=True, exist_ok=True)

        scenario_rows = []
        for scenario_filename, manifest_path in zip(playlist, scenario_manifests, strict=False):
            row = {
                "scenario_filename": scenario_filename,
                "scenario_name": Path(scenario_filename).stem,
                "manifest_path": str(manifest_path),
            }
            try:
                manifest = _read_json(manifest_path)
                row["session_csv"] = str(manifest.get("paths", {}).get("session_csv", ""))
                row["lsl_enabled"] = bool(manifest.get("lsl_enabled"))
            except Exception:
                # Best-effort; don't crash an attended run if a manifest is malformed.
                pass
            scenario_rows.append(row)

        timestamp_tag = datetime.now().strftime("%Y%m%dT%H%M%S")
        run_manifest_path = run_manifest_dir / f"run_manifest_{timestamp_tag}.json"
        if run_manifest_path.exists():
            print(f"ERROR: Refusing to overwrite existing run manifest: {run_manifest_path}", file=sys.stderr)
            return 2

        run_manifest = {
            "schema": "pilot_run_manifest_v0",
            "created_at": datetime.now().isoformat(),
            "participant": participant,
            "session": session,
            "seq_id": seq_id,
            "mode": {
                "pilot1": bool(args.pilot1),
                "calibration_only": bool(args.calibration_only),
                "dry_run": bool(args.dry_run),
                "verification": bool(args.verification),
            },
            "repo": {
                "commit": repo_commit,
                "openmatb_submodule_commit": submodule_commit,
            },
            "openmatb": {
                "output_root": str(output_root_path),
                "output_subdir": str(Path("openmatb") / participant / session),
                "session_root": str(session_root),
                "openmatb_dir": str(openmatb_dir),
            },
            "physiology": {
                "xdf_path": str(args.xdf_path) if args.xdf_path else None,
                "recording_scope": "calibration_only",
                "expected_streams": {
                    "markers": {"name": "OpenMATB", "type": "Markers"},
                    "eda": {"type": "EDA"},
                },
            },
            "playlist": scenario_rows,
            "qc": {
                "xdf_alignment": {
                    "required": bool(args.pilot1),
                    "skipped": bool(args.skip_xdf_qc),
                    "status": "pending" if (args.pilot1 and not args.skip_xdf_qc) else ("skipped" if args.skip_xdf_qc else "not_required"),
                    "report_path": None,
                    "thresholds": {
                        "median_abs_ms": 20,
                        "p95_abs_ms": 50,
                        "drift_ms_per_min": 5,
                        "hard_fail_drift_ms_per_min": 20,
                    },
                }
            },
        }

        _atomic_write_json(run_manifest_path, run_manifest)
        print(f"Wrote run manifest: {run_manifest_path}")
    except Exception as exc:
        print(f"ERROR: Failed to write run-level manifest: {exc}", file=sys.stderr)
        return 2
    
    # Run XDF↔CSV marker alignment QC if --pilot1 and not --skip-xdf-qc
    if args.pilot1 and not args.skip_xdf_qc and args.xdf_path:
        print("\n" + "="*60)
        print("Running XDF↔CSV marker alignment QC...")
        print("="*60)
        
        try:
            # Import from sibling verification module
            import importlib.util
            qc_script = Path(__file__).parent / "verification" / "verify_xdf_alignment.py"
            spec = importlib.util.spec_from_file_location("verify_xdf_alignment", qc_script)
            verify_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verify_module)
            run_qc = verify_module.run_qc
            
            csv_paths = [Path(row["session_csv"]) for row in scenario_rows if row.get("session_csv")]
            qc_report_path = run_manifest_path.with_suffix(".qc_alignment.json")
            
            qc_report = run_qc(
                xdf_path=Path(args.xdf_path),
                csv_paths=csv_paths,
                thresholds=run_manifest["qc"]["xdf_alignment"]["thresholds"],
                output_path=qc_report_path,
            )
            
            qc_passed = qc_report.get("results", {}).get("passed", False)
            run_manifest["qc"]["xdf_alignment"]["status"] = "passed" if qc_passed else "failed"
            run_manifest["qc"]["xdf_alignment"]["report_path"] = str(qc_report_path)
            
            # Update run manifest with QC results
            _atomic_write_json(run_manifest_path, run_manifest)
            
            if not qc_passed:
                print("\n⚠️  XDF alignment QC FAILED")
                print("Fail reasons:")
                for reason in qc_report.get("results", {}).get("fail_reasons", []):
                    print(f"  - {reason}")
                print(f"\nSee QC report: {qc_report_path}")
            else:
                print("\n✓ XDF alignment QC PASSED")
                timing = qc_report.get("results", {}).get("timing", {})
                print(f"  Median error: {timing.get('median_abs_error_ms', 0):.1f} ms")
                print(f"  95th pct:     {timing.get('p95_abs_error_ms', 0):.1f} ms")
        
        except ImportError as exc:
            print(f"WARNING: Could not run XDF QC (missing dependency): {exc}", file=sys.stderr)
            print("Install with: pip install pyxdf", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = "skipped_missing_dep"
            _atomic_write_json(run_manifest_path, run_manifest)
        except Exception as exc:
            print(f"WARNING: XDF QC failed with error: {exc}", file=sys.stderr)
            run_manifest["qc"]["xdf_alignment"]["status"] = f"error: {exc}"
            _atomic_write_json(run_manifest_path, run_manifest)
    
    # Update participant assignments
    if not args.skip_assignment_update:
        # Ensure participant is in assignments
        if participant not in participants_data:
            participants_data[participant] = {
                "sequence": seq_id,
                "sessions_completed": [],
                "last_run": None
            }
        
        # Add session to completed list if not already there
        pdata = participants_data[participant]
        if session not in pdata.get("sessions_completed", []):
            if "sessions_completed" not in pdata:
                pdata["sessions_completed"] = []
            pdata["sessions_completed"].append(session)
        
        # Update last run timestamp
        pdata["last_run"] = datetime.now().isoformat()
        
        # Save assignments
        assignments["participants"] = participants_data
        _save_assignments(repo_root, assignments, dry_run=args.skip_assignment_update)
        print(f"Updated participant assignments: {participant} completed {session}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
