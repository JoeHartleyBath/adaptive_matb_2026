"""Control LabRecorder (XDF) via its Remote Control Socket (RCS).

Purpose
- Start/stop LabRecorder recordings without interacting with the GUI.
- Set a BIDS-like output path template and session identifiers.

This script assumes:
- LabRecorder is already running.
- RCS is enabled (RCSEnabled=1) and listening on a TCP port (default 22345).

References
- LabRecorder RCS commands and filename options are documented in:
  https://github.com/labstreaminglayer/App-LabRecorder

Notes
- LabRecorder typically does not send responses to these commands; we operate
  best-effort and fail fast on connection errors.
- If the target XDF already exists, LabRecorder will rename the existing file
  (e.g., appending _oldX) to prevent overwriting.
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path


DEFAULT_RCS_HOST = "127.0.0.1"
DEFAULT_RCS_PORT = 22345

# A pragmatic BIDS-like default for multi-modal physiology stored as a single XDF.
# Placeholders are LabRecorder placeholders.
DEFAULT_TEMPLATE = (
    "sub-%p\\ses-%s\\physio\\sub-%p_ses-%s_task-%b_acq-%a_%m.xdf"
)


def _send_lines(host: str, port: int, lines: list[str], timeout_s: float = 3.0) -> None:
    data = "".join(f"{line.strip()}\n" for line in lines if line.strip()).encode("utf-8")

    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall(data)


def _build_filename_command(
    root: Path,
    template: str,
    participant: str | None,
    session: str | None,
    task: str | None,
    acquisition: str | None,
    modality: str | None,
    run: str | None,
) -> str:
    parts: list[str] = ["filename"]
    parts.append(f"{{root:{str(root)}}}")
    parts.append(f"{{template:{template}}}")

    if task:
        parts.append(f"{{task:{task}}}")
    if run:
        parts.append(f"{{run:{run}}}")
    if participant:
        parts.append(f"{{participant:{participant}}}")
    if session:
        parts.append(f"{{session:{session}}}")
    if acquisition:
        parts.append(f"{{acquisition:{acquisition}}}")
    if modality:
        parts.append(f"{{modality:{modality}}}")

    return " ".join(parts)


def _expected_xdf_path(
    root: Path,
    template: str,
    participant: str,
    session: str,
    task: str,
    acquisition: str,
    modality: str,
    run: str | None,
) -> Path:
    # We compute an *expected* path by applying the same placeholder rules used
    # by LabRecorder. This is for operator convenience and for passing an
    # anticipated path to the runner (e.g., --xdf-path). LabRecorder may still
    # rename the file if it already exists.
    mapped = template
    mapped = mapped.replace("%p", participant)
    mapped = mapped.replace("%s", session)
    mapped = mapped.replace("%b", task)
    mapped = mapped.replace("%a", acquisition)
    mapped = mapped.replace("%m", modality)
    if run is not None:
        mapped = mapped.replace("%r", str(run))
        mapped = mapped.replace("%n", str(run))

    mapped = mapped.replace("/", "\\")
    return (root / mapped).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Control LabRecorder via RCS (TCP)")
    parser.add_argument("--host", default=DEFAULT_RCS_HOST, help="RCS host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_RCS_PORT, help="RCS port (default: 22345)")
    parser.add_argument("--timeout", type=float, default=3.0, help="Socket timeout in seconds")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_update = sub.add_parser("update", help="Refresh stream list")
    p_update.add_argument("--select-all", action="store_true", help="Also send 'select all'")

    sub.add_parser("select-all", help="Select all streams")
    sub.add_parser("select-none", help="Select no streams")
    sub.add_parser("start", help="Start recording")
    sub.add_parser("stop", help="Stop recording")

    p_start_bids = sub.add_parser("start-bids", help="Set filename (BIDS-like) + select all + start")
    p_start_bids.add_argument("--root", default=r"C:\\data\\adaptive_matb\\physiology", help="Recording root directory")
    p_start_bids.add_argument("--template", default=DEFAULT_TEMPLATE, help="Path template relative to root")
    p_start_bids.add_argument("--participant", required=True, help="Participant label (e.g., P008)")
    p_start_bids.add_argument("--session", required=True, help="Session label (e.g., S001)")
    p_start_bids.add_argument("--task", default="matb", help="Task/block label for %%b (default: matb)")
    p_start_bids.add_argument("--acq", default="pilot1", help="Acquisition label for %%a (default: pilot1)")
    p_start_bids.add_argument("--modality", default="physio", help="Modality label for %%m (default: physio)")
    p_start_bids.add_argument(
        "--run",
        default=None,
        help="Run index (maps to %%r and %%n when used in template). Note: LabRecorder RCS 'run' may not work in some versions.",
    )
    p_start_bids.add_argument(
        "--print-expected-path",
        action="store_true",
        help="Print the expected .xdf path (useful for --xdf-path).",
    )

    args = parser.parse_args()

    host: str = args.host
    port: int = args.port

    if args.cmd == "update":
        lines = ["update"]
        if args.select_all:
            lines.append("select all")
        _send_lines(host, port, lines, timeout_s=float(args.timeout))
        return 0

    if args.cmd == "select-all":
        _send_lines(host, port, ["select all"], timeout_s=float(args.timeout))
        return 0

    if args.cmd == "select-none":
        _send_lines(host, port, ["select none"], timeout_s=float(args.timeout))
        return 0

    if args.cmd == "start":
        _send_lines(host, port, ["start"], timeout_s=float(args.timeout))
        return 0

    if args.cmd == "stop":
        _send_lines(host, port, ["stop"], timeout_s=float(args.timeout))
        return 0

    if args.cmd == "start-bids":
        root = Path(str(args.root))
        template = str(args.template)

        filename_cmd = _build_filename_command(
            root=root,
            template=template,
            participant=str(args.participant),
            session=str(args.session),
            task=str(args.task),
            acquisition=str(args.acq),
            modality=str(args.modality),
            run=str(args.run) if args.run is not None else None,
        )

        if args.print_expected_path:
            expected = _expected_xdf_path(
                root=root,
                template=template,
                participant=str(args.participant),
                session=str(args.session),
                task=str(args.task),
                acquisition=str(args.acq),
                modality=str(args.modality),
                run=str(args.run) if args.run is not None else None,
            )
            print(str(expected))

        # Typical robust sequence: update → select all → exclude TRG → set filename → start.
        # The eego amplifier exposes a separate LSL outlet (type="TRG") for each
        # amp's trigger channel.  It is not needed for EEG analysis; deselect it.
        _send_lines(
            host,
            port,
            [
                "update",
                "select all",
                "deselect {type:TRG}",
                filename_cmd,
                "start",
            ],
            timeout_s=float(args.timeout),
        )
        return 0

    raise AssertionError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
