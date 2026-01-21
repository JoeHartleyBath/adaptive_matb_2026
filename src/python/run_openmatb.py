"""Launch OpenMATB with repo-safe output paths.

This wrapper enforces that participant/session identifiers are provided and configures
OpenMATB to write session logs outside the git repo.

Usage (PowerShell):
  cd src/python/vendor/openmatb
  .\.venv\Scripts\Activate.ps1
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
import os
import re
import subprocess
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

    args = parser.parse_args()

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

    os.environ["OPENMATB_OUTPUT_ROOT"] = str(output_root_path)
    os.environ["OPENMATB_OUTPUT_SUBDIR"] = str(Path("openmatb") / participant / session)

    # Preserve for downstream tooling, even if OpenMATB itself doesn't use them.
    os.environ["OPENMATB_PARTICIPANT"] = participant
    os.environ["OPENMATB_SESSION"] = session

    return subprocess.call([sys.executable, "main.py"], cwd=str(openmatb_dir))


if __name__ == "__main__":
    raise SystemExit(main())
