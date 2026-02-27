"""Record selected LSL streams to newline-delimited JSON.

This is a lightweight Python recorder intended to replace LabRecorder for
Pilot 1 workflows where we only need reproducible stream capture and marker
alignment QC inputs.

Output format (.jsonl):
- First line: session header (type="header")
- Stream announcements: type="stream_info"
- Samples: type="sample"
- Final line: type="footer"
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any


try:
    import pylsl
except ImportError as exc:
    print("ERROR: pylsl is required. Install with: pip install pylsl", file=sys.stderr)
    raise SystemExit(2) from exc


_RUNNING = True


def _signal_handler(signum, _frame) -> None:
    global _RUNNING
    _RUNNING = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        return float(value)
    except Exception:
        return str(value)


def _stream_matches(info: pylsl.StreamInfo, include_names: set[str], include_types: set[str]) -> bool:
    name = (info.name() or "").strip()
    stream_type = (info.type() or "").strip()

    if include_names and name in include_names:
        return True
    if include_types and stream_type in include_types:
        return True

    return not include_names and not include_types


def _stream_key(info: pylsl.StreamInfo) -> str:
    source = info.source_id() or ""
    return f"{info.name()}::{info.type()}::{source}::{info.channel_count()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Record selected LSL streams to JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--include-name",
        action="append",
        default=[],
        help="Record streams with this exact name (repeatable)",
    )
    parser.add_argument(
        "--include-type",
        action="append",
        default=[],
        help="Record streams with this exact type (repeatable)",
    )
    parser.add_argument(
        "--resolve-interval",
        type=float,
        default=1.0,
        help="Seconds between stream discovery scans",
    )
    parser.add_argument(
        "--max-samples-per-pull",
        type=int,
        default=64,
        help="Maximum samples pulled per stream per loop",
    )

    args = parser.parse_args()

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_names = {name.strip() for name in args.include_name if name and name.strip()}
    include_types = {stream_type.strip() for stream_type in args.include_type if stream_type and stream_type.strip()}

    print(f"Recording LSL streams to: {output_path}")
    if include_names:
        print(f"  include names: {sorted(include_names)}")
    if include_types:
        print(f"  include types: {sorted(include_types)}")

    inlets: dict[str, pylsl.StreamInlet] = {}
    stream_meta: dict[str, dict[str, Any]] = {}
    sample_counts: dict[str, int] = {}

    start_wall = datetime.now().isoformat()
    start_lsl = float(pylsl.local_clock())

    with open(output_path, "w", encoding="utf-8", buffering=1) as out_f:
        out_f.write(
            json.dumps(
                {
                    "type": "header",
                    "schema": "lsl_recording_v0",
                    "created_at": start_wall,
                    "lsl_start_time": start_lsl,
                    "include_names": sorted(include_names),
                    "include_types": sorted(include_types),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        last_resolve = 0.0

        while _RUNNING:
            now = time.time()
            if now - last_resolve >= max(0.1, args.resolve_interval):
                last_resolve = now
                try:
                    discovered = pylsl.resolve_streams(wait_time=0.2)
                except Exception:
                    discovered = []

                for info in discovered:
                    if not _stream_matches(info, include_names, include_types):
                        continue

                    key = _stream_key(info)
                    if key in inlets:
                        continue

                    try:
                        inlet = pylsl.StreamInlet(info, max_buflen=120)
                    except Exception:
                        continue

                    inlets[key] = inlet
                    stream_meta[key] = {
                        "name": info.name(),
                        "type": info.type(),
                        "source_id": info.source_id(),
                        "channel_count": int(info.channel_count()),
                        "nominal_srate": float(info.nominal_srate()),
                    }
                    sample_counts[key] = 0

                    out_f.write(
                        json.dumps(
                            {
                                "type": "stream_info",
                                "stream_key": key,
                                "info": stream_meta[key],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    print(f"Attached stream: {stream_meta[key]['name']} ({stream_meta[key]['type']})")

            for key, inlet in list(inlets.items()):
                try:
                    samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=args.max_samples_per_pull)
                except Exception:
                    continue

                if not timestamps:
                    continue

                for sample, ts in zip(samples, timestamps):
                    out_f.write(
                        json.dumps(
                            {
                                "type": "sample",
                                "stream_key": key,
                                "timestamp": float(ts),
                                "sample": _to_jsonable(sample),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    sample_counts[key] += 1

            time.sleep(0.01)

        end_lsl = float(pylsl.local_clock())
        out_f.write(
            json.dumps(
                {
                    "type": "footer",
                    "ended_at": datetime.now().isoformat(),
                    "lsl_end_time": end_lsl,
                    "duration_s": end_lsl - start_lsl,
                    "sample_counts": sample_counts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print("LSL recorder stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
