"""Launch OpenMATB with repo-safe output paths.

This wrapper enforces that participant/session identifiers are provided and configures
OpenMATB to write session logs outside the git repo.

Usage (PowerShell):
  cd src/python/vendor/openmatb
    ./.venv/Scripts/Activate.ps1
  python ..\..\run_openmatb.py --participant P001 --session S001

Environment variables (optional):
  OPENMATB_OUTPUT_ROOT   (default: C:\data\adaptive_matb)
  OPENMATB_PARTICIPANT / OPENMATB_PARTICIPANT_ID
  OPENMATB_SESSION / OPENMATB_SESSION_ID

OpenMATB uses:
  OPENMATB_OUTPUT_ROOT and OPENMATB_OUTPUT_SUBDIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
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
    unattended: bool,
    dry_run: bool,
    scenario_filename: str,
    abort_reason: Optional[str] = None,
) -> None:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["seq_id"] = seq_id
    manifest["unattended"] = unattended
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
        "pilot_familiarisation.txt",
        "pilot_training_T1_LOW.txt",
        "pilot_training_T2_MODERATE.txt",
        "pilot_training_T3_HIGH.txt",
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

    if seq_id not in retained_levels:
        raise ValueError(f"Unknown sequence ID: {seq_id}")

    levels = retained_levels[seq_id]
    for level in levels:
        playlist.append(f"pilot_retained_{level}.txt")

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
    scenario_target_path = openmatb_dir / "includes" / "scenarios" / scenario_filename
    if not scenario_target_path.exists():
        print(f"Scenario file not found: {scenario_target_path}", file=sys.stderr)
        return 2

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
    env["OPENMATB_UNATTENDED"] = "1" if args.unattended else "0"

    if args.dry_run_timeout_seconds is not None and args.unattended:
        env["OPENMATB_DRY_RUN_TIMEOUT_SECONDS"] = str(args.dry_run_timeout_seconds)

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
        "        if os.environ.get('OPENMATB_UNATTENDED') == '1':\n"
        "            self.modal_dialog.on_delete()\n"
        "Window.display_session_id = _display_session_id\n"
        "if os.environ.get('OPENMATB_UNATTENDED') == '1':\n"
        "    from core.logger import logger\n"
        "    import threading, time as _time\n"
        "    from plugins.abstractplugin import BlockingPlugin\n"
        "    from plugins.genericscales import Genericscales\n"
        "    from plugins.instructions import Instructions\n"
        "    from plugins.labstreaminglayer import Labstreaminglayer\n"
        "    import core.scenario as scenario_mod\n"
        "    _orig_start = BlockingPlugin.start\n"
        "    def _patched_start(self, *args, **kwargs):\n"
        "        _orig_start(self, *args, **kwargs)\n"
        "        if isinstance(self, Labstreaminglayer):\n"
        "            return\n"
        "        if isinstance(self, (Instructions, Genericscales)):\n"
        "            logger.log_manual_entry(f'unattended_skip:{self.alias}', key='unattended')\n"
        "            self.blocking = False\n"
        "            self.stop()\n"
        "    BlockingPlugin.start = _patched_start\n"
        "    _orig_scenario_init = scenario_mod.Scenario.__init__\n"
        "    def _patched_scenario_init(self, contents=None):\n"
        "        _orig_scenario_init(self, contents)\n"
        "        for name, plugin in self.plugins.items():\n"
        "            if name != 'communications' and 'automaticsolver' in getattr(plugin, 'parameters', {}):\n"
        "                plugin.set_parameter('automaticsolver', True)\n"
        "        logger.log_manual_entry('unattended:automaticsolver_enabled_except_comms', key='unattended')\n"
        "    scenario_mod.Scenario.__init__ = _patched_scenario_init\n"
        "    timeout_raw = os.environ.get('OPENMATB_DRY_RUN_TIMEOUT_SECONDS', '')\n"
        "    timeout_seconds = int(timeout_raw) if timeout_raw else 0\n"
        "    if timeout_seconds > 0:\n"
        "        def _timeout_watchdog():\n"
        "            start = _time.monotonic()\n"
        "            while _time.monotonic() - start < timeout_seconds:\n"
        "                _time.sleep(0.2)\n"
        "            try:\n"
        "                logger.log_manual_entry(f'DRY RUN TIMEOUT: aborting after {timeout_seconds}s', key='abort')\n"
        "            except Exception:\n"
        "                pass\n"
        "            try:\n"
        "                if Window.MainWindow is not None:\n"
        "                    Window.MainWindow.alive = False\n"
        "            except Exception:\n"
        "                pass\n"
        "        threading.Thread(target=_timeout_watchdog, daemon=True).start()\n"
        "runpy.run_path('main.py', run_name='__main__')\n"
    )

    timeout_seconds = args.dry_run_timeout_seconds if args.unattended else None
    timeout_triggered = False
    start_time = time.monotonic()

    # Pass the modified environment
    proc = subprocess.Popen([sys.executable, "-c", bootstrap], cwd=str(openmatb_dir), env=env)

    if timeout_seconds is None:
        exit_code = proc.wait()
    else:
        grace_seconds = 3
        deadline = None
        exit_code = None
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                break
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_seconds and not timeout_triggered:
                timeout_triggered = True
                deadline = time.monotonic() + grace_seconds
                print(f"TIMEOUT: aborting {scenario_filename} after {timeout_seconds}s")
            if timeout_triggered and deadline is not None and time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=grace_seconds)
                except subprocess.TimeoutExpired:
                    proc.kill()
                exit_code = proc.poll()
                if exit_code is None:
                    exit_code = 3
                break
            time.sleep(0.2)
        if timeout_triggered and exit_code == 0:
            exit_code = 3

    # Post-process manifests
    if exit_code != 0:
        manifests_after = _list_manifest_paths(sessions_dir)
        new_manifests = sorted(manifests_after - manifests_before)
        if len(new_manifests) == 1:
            _write_seq_id_into_manifest(
                new_manifests[0],
                seq_id=seq_id,
                unattended=args.unattended,
                dry_run=args.dry_run,
                scenario_filename=scenario_filename,
                abort_reason="timeout" if timeout_triggered else (f"exit_code_{exit_code}"),
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

    _write_seq_id_into_manifest(
        new_manifests[0],
        seq_id=seq_id,
        unattended=args.unattended,
        dry_run=args.dry_run,
        scenario_filename=scenario_filename,
        abort_reason="timeout" if timeout_triggered else None,
    )
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
        required=True,
        choices=("SEQ1", "SEQ2", "SEQ3"),
        help="Retained-order sequence ID (required): SEQ1, SEQ2, or SEQ3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic dry-run scenario artifact.",
    )
    parser.add_argument(
        "--full-session",
        action="store_true",
        help="Explicitly allow running the full session when unattended.",
    )
    parser.add_argument(
        "--unattended",
        action="store_true",
        help="Run without any user input (skip instructions/TLX, enable automation).",
    )
    parser.add_argument(
        "--dry-run-timeout-seconds",
        type=int,
        default=None,
        help="Wall-clock timeout (seconds) for unattended runs only.",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=1,
        help="Fast-forward speed multiplier (e.g. 20) for unattended/testing runs.",
    )

    args = parser.parse_args()

    if args.dry_run and args.full_session:
        print("Use only one of --dry-run or --full-session", file=sys.stderr)
        return 2

    if args.unattended and not args.dry_run and not args.full_session:
        print("WARNING: Unattended run without --full-session; defaulting to --dry-run.")
        args.dry_run = True

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

    if args.unattended:
        print("WARNING: Unattended mode is ON. Inputs, instructions, and TLX will be auto-skipped.")
        if args.dry_run_timeout_seconds:
            if args.dry_run_timeout_seconds <= 0:
               print("--dry-run-timeout-seconds must be a positive integer", file=sys.stderr)
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
        playlist = _get_playlist(args.seq_id, args.dry_run)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Running sequence: {args.seq_id}")
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
            seq_id=args.seq_id,
            args=args,
            repo_commit=repo_commit,
            submodule_commit=submodule_commit,
        )

        if exit_code != 0:
            print(f"\n!!! Scenario {scenario_filename} failed (code {exit_code}). Stopping sequence. !!!", file=sys.stderr)
            return exit_code
        
        # Simple separation between blocks
        print(f"Scenario {scenario_filename} completed successfully.")
        if not args.unattended:
            # Maybe a small pause or log message?
            pass

    print("\nAll scenarios in playlist completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
