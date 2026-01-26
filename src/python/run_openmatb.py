"""Launch OpenMATB with repo-safe output paths.

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
import json
import os
import re
import subprocess
import shutil
import sys
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


def _get_playlist(seq_id: str, dry_run: bool) -> list[str]:
    if dry_run:
        return ["pilot_dry_run_v0.txt"]

    # Fixed training sequence
    playlist = [
        "pilot_practice_intro.txt",
        "pilot_practice_low.txt",
        "pilot_practice_moderate.txt",
        "pilot_practice_high.txt",
    ]

    # Retained blocks based on counterbalancing sequence
    # SEQ1: Low -> Moderate -> High
    # SEQ2: Moderate -> High -> Low
    # SEQ3: High -> Low -> Moderate
    retained_levels = {
        "SEQ1": ["LOW", "MODERATE", "HIGH"],
        "SEQ2": ["MODERATE", "HIGH", "LOW"],
        "SEQ3": ["HIGH", "LOW", "MODERATE"],
    }

    try:
        levels = retained_levels[seq_id]
    except KeyError as exc:
        raise ValueError(f"Unknown sequence ID: {seq_id}") from exc
    for level in levels:
        playlist.append(f"pilot_static_{level.lower()}.txt")

    return playlist


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
) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scenario_source_path = repo_root / "scenarios" / scenario_filename
    if not scenario_source_path.exists():
        print(f"Scenario file not found: {scenario_source_path}", file=sys.stderr)
        return 2

    scenario_target_path = openmatb_dir / "includes" / "scenarios" / scenario_filename
    scenario_target_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_target_path.write_text(scenario_source_path.read_text(encoding="utf-8"), encoding="utf-8")

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
        return exit_code

    manifests_after = _list_manifest_paths(sessions_dir)
    new_manifests = sorted(manifests_after - manifests_before)
    if len(new_manifests) != 1:
        print(
            f"WARNING: Expected 1 new manifest for {scenario_filename}, found {len(new_manifests)} in {sessions_dir}. "
            "Skipping manifest metadata injection.",
            file=sys.stderr,
        )
        return 2

    manifest_path = new_manifests[0]
    _write_seq_id_into_manifest(
        manifest_path,
        seq_id=seq_id,
        dry_run=args.dry_run,
        scenario_filename=scenario_filename,
        abort_reason=None,
    )

    # If OpenMATB reports scenario parsing/runtime errors, treat this run as failed.
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        errors_log = Path(manifest.get("paths", {}).get("scenario_errors_log", ""))
        if errors_log and errors_log.exists() and errors_log.stat().st_size > 0:
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
            return 2

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
                return 2
    except Exception:
        # If we can't read the manifest or error log, do not crash the runner.
        pass
    return 0


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
        help="Retained-order sequence ID (SEQ1/SEQ2/SEQ3). Can also be set via OPENMATB_SEQ_ID.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic dry-run scenario artifact.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Fast-forward speed multiplier (e.g. 5) for testing runs.",
    )

    args = parser.parse_args()

    seq_id = args.seq_id or _get_env_first("OPENMATB_SEQ_ID")
    if not seq_id:
        if args.dry_run:
            seq_id = "DRYRUN"
        else:
            print(
                "Missing required --seq-id (SEQ1/SEQ2/SEQ3). Provide --seq-id or set OPENMATB_SEQ_ID.",
                file=sys.stderr,
            )
            return 2

    participant_raw = args.participant or _get_env_first("OPENMATB_PARTICIPANT", "OPENMATB_PARTICIPANT_ID")
    session_raw = args.session or _get_env_first("OPENMATB_SESSION", "OPENMATB_SESSION_ID")

    if participant_raw is None or session_raw is None:
        missing = []
        if participant_raw is None:
            missing.append("participant")
        if session_raw is None:
            missing.append("session")
        msg = (
            "Missing required identifiers: "
            + ", ".join(missing)
            + ". Provide --participant/--session or set OPENMATB_PARTICIPANT and OPENMATB_SESSION."
        )
        print(msg, file=sys.stderr)
        return 2

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
        playlist = _get_playlist(seq_id, args.dry_run)
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

    for scenario_filename in playlist:
        exit_code = _run_single_scenario(
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
        
        # Simple separation between blocks
        print(f"Scenario {scenario_filename} completed successfully.")
        # (Interactive UI and blocking dialogs are expected in attended mode)

    print("\nAll scenarios in playlist completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
