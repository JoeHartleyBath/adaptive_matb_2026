"""Unified pilot verification harness.

This script is the canonical "run + verify" entrypoint for the pilot playlist.

It is designed to work with the current behaviour of `src/python/run_openmatb.py`,
which runs a sequence as a playlist of scenario files (one OpenMATB run per file)
and writes one manifest+CSV per scenario.

What it does:
- Runs the existing static verifier (`verify_pilot_scenarios.py`) for repo artifacts.
- Optionally launches the OpenMATB runner for a given participant/session/SEQ.
- Collects the produced manifests/CSVs and verifies:
  - manifests match expected scenario names and identifiers
  - CSV markers match the scenario-defined marker payloads with tokens substituted
  - segment durations in CSV match intended marker timing (within tolerance)
  - per-segment event counts match expected counts (LEVEL_COUNTS)
  - comms radioprompt timestamps match intended schedule (within tolerance)

Usage:
    python src/python/verify_pilot.py --participant P001 --session S001 --seq-id SEQ1

Notes:
- This is attended/interactive; OpenMATB will display UI.
- Fast-forward is controlled via `--speed` and is wired into OpenMATB clock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# Ensure we can import runner/util modules when executed from repo root.
# File is located at <repo>/src/python/verification/verify_pilot.py.
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PYTHON = REPO_ROOT / "src" / "python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))


# We reuse verifier utilities (parsers + constants) but do NOT reuse unattended-only assumptions.
from verification import verify_pilot_scenarios as vps


@dataclass
class CheckFailure:
    label: str
    details: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_calibration_verifier(python: str) -> int:
    """Run the repo-level static checks (scenarios/assets/contracts)."""
    proc = subprocess.run(
        [python, str(_repo_root() / "src" / "python" / "verification" / "verify_pilot_scenarios.py")]
    )
    return int(proc.returncode)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text_lines(path: Path, *, max_lines: int = 50) -> list[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines: list[str] = []
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line.rstrip("\n"))
            return lines
    except Exception as exc:
        return [f"(unable to read {path}: {exc})"]


def _session_dir(output_root: Path, participant: str, session: str) -> Path:
    return output_root / "openmatb" / participant / session / "sessions"


def _list_manifests(session_dir: Path) -> list[Path]:
    if not session_dir.exists():
        return []
    return sorted(session_dir.glob("**/*.manifest.json"), key=lambda p: p.stat().st_mtime)


def _intended_segments_from_scenario_events(events: list[vps.Event]) -> dict[str, vps.Segment]:
    marker_events = vps._find_marker_events(events)
    by_name: dict[str, vps.Event] = {}
    for ev in marker_events:
        name = vps._marker_name(ev.command[-1])
        if name:
            by_name[name] = ev

    segments: dict[str, vps.Segment] = {}

    def add_segment(label: str, start_marker: str, end_marker: str) -> None:
        if start_marker not in by_name or end_marker not in by_name:
            return
        start = by_name[start_marker]
        end = by_name[end_marker]
        segments[label] = vps.Segment(
            label=label,
            start_sec=float(start.time_sec),
            end_sec=float(end.time_sec),
            duration_sec=float(end.time_sec - start.time_sec),
            start_line=start.line_no,
            end_line=end.line_no,
        )

    add_segment("T1", "TRAINING/T1/START", "TRAINING/T1/END")
    add_segment("T2", "TRAINING/T2/START", "TRAINING/T2/END")
    add_segment("T3", "TRAINING/T3/START", "TRAINING/T3/END")

    for level in ("LOW", "MODERATE", "HIGH"):
        add_segment(f"calibration_{level}", f"calibration/{level}/START", f"calibration/{level}/END")

    # Current scenarios emit TLX markers keyed to the immediately preceding calibration level,
    # e.g., TLX/calibration_LOW/START and TLX/calibration_LOW/END.
    for level in ("LOW", "MODERATE", "HIGH"):
        add_segment(
            f"TLX_calibration_{level}",
            f"TLX/calibration_{level}/START",
            f"TLX/calibration_{level}/END",
        )

    return segments


def _level_for_segment(label: str) -> Optional[str]:
    if label == "T1":
        return "LOW"
    if label == "T2":
        return "MODERATE"
    if label == "T3":
        return "HIGH"
    if label.startswith("calibration_"):
        level = label.replace("calibration_", "")
        return level if level in vps.ALLOWED_LEVELS else None
    return None


def _count_events_in_window(csv_events: list[vps.CsvEvent], start_sec: float, end_sec: float) -> dict[str, int]:
    def is_sysmon(ev: vps.CsvEvent) -> bool:
        return ev.module == "sysmon" and "failure" in ev.address and ev.value.lower() in {"true", "1"}

    def is_comm(ev: vps.CsvEvent) -> bool:
        return ev.module == "communications" and ev.address == "radioprompt"

    def is_resman(ev: vps.CsvEvent) -> bool:
        return ev.module == "resman" and "pump" in ev.address and ev.value == "failure"

    relevant = [e for e in csv_events if start_sec <= e.scenario_time < end_sec]
    return {
        "sysmon": sum(1 for e in relevant if is_sysmon(e)),
        "communications": sum(1 for e in relevant if is_comm(e)),
        "resman": sum(1 for e in relevant if is_resman(e)),
    }


def _check_monotonic_scenario_time(csv_events: list[vps.CsvEvent], csv_path: Path) -> Optional[CheckFailure]:
    prev = None
    for ev in csv_events:
        if prev is not None and ev.scenario_time < prev:
            return CheckFailure(
                label="scenario_time not monotonic",
                details=[f"{csv_path}: row {ev.row_index} has {ev.scenario_time} < {prev}"],
            )
        prev = ev.scenario_time
    return None


def _expected_marker_payloads_for_scenario(
    scenario_path: Path,
    participant: str,
    session: str,
    seq_id: str,
) -> list[str]:
    events = vps._parse_events(scenario_path)
    payloads: list[str] = []
    for ev in vps._find_marker_events(events):
        payload = ev.command[-1]
        payloads.append(
            payload.replace(vps.TOKEN_PID, participant)
            .replace(vps.TOKEN_SID, session)
            .replace(vps.TOKEN_SEQ, seq_id)
        )
    return payloads


def _check_markers_match_scenario(
    *,
    scenario_path: Path,
    csv_path: Path,
    participant: str,
    session: str,
    seq_id: str,
) -> Optional[CheckFailure]:
    expected = _expected_marker_payloads_for_scenario(scenario_path, participant, session, seq_id)
    observed = vps._extract_markers_from_csv(csv_path)

    missing = [m for m in expected if m not in observed]
    token_leaks = [m for m in observed if vps.TOKEN_PID in m or vps.TOKEN_SID in m or vps.TOKEN_SEQ in m]

    bad_id = []
    needle = f"|pid={participant}|sid={session}|seq={seq_id}"
    for m in observed:
        if m.startswith("STUDY/V0/") and "|" in m and needle not in m:
            bad_id.append(m)

    if missing or token_leaks or bad_id:
        details: list[str] = []
        if missing:
            details.append(f"Missing markers ({len(missing)}):")
            details.extend(missing[:10])
        if token_leaks:
            details.append(f"Token leaks in CSV markers ({len(token_leaks)}):")
            details.extend(token_leaks[:10])
        if bad_id:
            details.append(f"Markers with wrong pid/sid/seq ({len(bad_id)}):")
            details.extend(bad_id[:10])
        return CheckFailure(label="CSV markers do not match scenario", details=details)

    return None


def _check_dynamic_segments(
    *,
    scenario_path: Path,
    csv_path: Path,
    tolerance_sec: float,
) -> tuple[list[str], list[CheckFailure]]:
    intended_events = vps._parse_events(scenario_path)
    intended = _intended_segments_from_scenario_events(intended_events)

    csv_events = vps._parse_csv_events(csv_path)
    observed = vps._collect_csv_segments(csv_events)

    # Only require segments the scenario actually defines.
    intended = {k: v for k, v in intended.items() if k in intended}

    lines: list[str] = []
    failures: list[CheckFailure] = []

    if intended:
        # Duration check
        duration_ev = vps._dynamic_duration_check(scenario_path.name, intended, observed, tolerance_sec, csv_path)
        lines.extend([e.label for e in duration_ev])

        # Convert duration table to pass/fail by recomputing deltas.
        for key, seg in intended.items():
            if key not in observed:
                failures.append(CheckFailure("Missing CSV segment", [f"{key} missing in {csv_path}"]))
                continue
            delta = observed[key].duration_sec - seg.duration_sec
            if abs(delta) > tolerance_sec:
                failures.append(
                    CheckFailure(
                        "Segment duration mismatch",
                        [f"{key}: intended={seg.duration_sec:.2f}s observed={observed[key].duration_sec:.2f}s delta={delta:.2f}s"],
                    )
                )

        # Count check (only for training/calibration segments)
        headers = ["Segment", "Level", "Sysmon", "Comms", "Resman", "Expected", "PASS"]
        rows: list[list[str]] = []
        for label, seg in intended.items():
            level = _level_for_segment(label)
            if not level:
                continue
            if label not in observed:
                continue
            counts = _count_events_in_window(csv_events, observed[label].start_sec, observed[label].end_sec)
            expected = vps.LEVEL_COUNTS.get(level, {})
            pass_seg = (
                expected
                and counts.get("sysmon", -1) == expected.get("sysmon", -2)
                and counts.get("communications", -1) == expected.get("communications", -2)
                and counts.get("resman", -1) == expected.get("resman", -2)
            )
            rows.append(
                [
                    label,
                    level,
                    str(counts.get("sysmon", 0)),
                    str(counts.get("communications", 0)),
                    str(counts.get("resman", 0)),
                    f"{expected.get('sysmon', '?')}/{expected.get('communications', '?')}/{expected.get('resman', '?')}",
                    "PASS" if pass_seg else "FAIL",
                ]
            )
            if not pass_seg:
                failures.append(
                    CheckFailure(
                        "Event count mismatch",
                        [
                            f"{label} ({level}) :: observed sysmon={counts.get('sysmon', 0)} comm={counts.get('communications', 0)} resman={counts.get('resman', 0)}",
                            f"expected sysmon={expected.get('sysmon')} comm={expected.get('communications')} resman={expected.get('resman')}",
                        ],
                    )
                )

        if rows:
            lines.append("Event counts (per segment):")
            lines.extend(vps._format_table(headers, rows))

        # Comms schedule check (compare intended comm timestamps vs observed)
        expected_comm_times = [float(ev.time_sec) for ev in intended_events if ev.plugin == "communications" and ev.command and ev.command[0] == "radioprompt"]
        if expected_comm_times:
            comm_ev = vps._dynamic_comm_schedule_check(
                scenario_path.name,
                expected_comm_times,
                csv_events,
                tolerance_sec,
                csv_path,
            )
            lines.extend([e.label for e in comm_ev])
            observed_comm_times = sorted([e.scenario_time for e in csv_events if e.module == "communications" and e.address == "radioprompt"])
            missing, extra = vps._match_expected_times(expected_comm_times, observed_comm_times, tolerance_sec)
            if missing or extra:
                failures.append(
                    CheckFailure(
                        "Comms schedule mismatch",
                        [
                            f"missing={missing[:10]}",
                            f"extra={extra[:10]}",
                        ],
                    )
                )

    else:
        lines.append(f"No TRAINING/calibration/TLX segments detected in {scenario_path.name}; skipping timing/count checks.")

    return lines, failures


def _find_manifest_for_scenario(manifests: list[Path], scenario_stem: str) -> list[Path]:
    matches: list[Path] = []
    for mp in manifests:
        try:
            manifest = _load_json(mp)
        except Exception:
            continue
        if manifest.get("scenario_name") == scenario_stem:
            matches.append(mp)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenMATB pilot playlist and verify outputs.")
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--seq-id", required=True, choices=("SEQ1", "SEQ2", "SEQ3"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--openmatb-dir", type=Path, default=None)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.5)
    parser.add_argument("--skip-static", action="store_true", help="Skip static repo artifact checks.")
    parser.add_argument("--skip-run", action="store_true", help="Do not launch OpenMATB; verify existing outputs only.")
    args = parser.parse_args()

    repo_root = _repo_root()
    output_root = args.output_root or Path(os.environ.get("OPENMATB_OUTPUT_ROOT", r"C:\data\adaptive_matb"))

    failures: list[CheckFailure] = []

    if not args.skip_static:
        rc = _run_calibration_verifier(sys.executable)
        if rc != 0:
            print("Static verification failed; fix static issues before running dynamic checks.", file=sys.stderr)
            return rc

    # Run the attended OpenMATB sequence (interactive UI)
    if not args.skip_run:
        cmd = [
            sys.executable,
            str(repo_root / "src" / "python" / "run_openmatb.py"),
            "--participant",
            args.participant,
            "--session",
            args.session,
            "--seq-id",
            args.seq_id,
            "--summarise-performance",
            "--verification",
            "--speed",
            str(args.speed),
            "--output-root",
            str(output_root),
        ]
        if args.openmatb_dir:
            cmd.extend(["--openmatb-dir", str(args.openmatb_dir)])

        print("Launching OpenMATB runner (attended)...")
        print("Command:")
        print("  " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"Runner exited with code {rc}", file=sys.stderr)
            return rc

    session_dir = _session_dir(output_root, args.participant, args.session)
    manifests = _list_manifests(session_dir)
    if not manifests:
        print(f"No manifests found in {session_dir}", file=sys.stderr)
        return 2

    # Determine expected playlist using the runner's own mapping.
    import run_openmatb as runner

    playlist = runner._get_playlist(args.seq_id, False)
    scenario_dir = repo_root / "scenarios"

    print("\nVerifying produced outputs...")
    print(f"Session dir: {session_dir}")
    print(f"Expected scenarios ({len(playlist)}):")
    for s in playlist:
        print(f" - {s}")

    for scenario_filename in playlist:
        scenario_path = scenario_dir / scenario_filename
        scenario_stem = Path(scenario_filename).stem

        if not scenario_path.exists():
            failures.append(CheckFailure("Missing scenario file", [str(scenario_path)]))
            continue

        matches = _find_manifest_for_scenario(manifests, scenario_stem)
        if len(matches) != 1:
            failures.append(
                CheckFailure(
                    "Manifest association failure",
                    [f"Expected 1 manifest for {scenario_stem}, found {len(matches)}"],
                )
            )
            continue

        manifest_path = matches[0]
        manifest = _load_json(manifest_path)

        abort_reason = manifest.get("abort_reason")
        if abort_reason:
            failures.append(
                CheckFailure(
                    "Scenario aborted (runner)",
                    [f"abort_reason={abort_reason}", str(manifest_path)],
                )
            )
            continue

        # Surface OpenMATB-reported scenario errors early; they typically explain
        # downstream missing markers/segments.
        errors_log_str = manifest.get("paths", {}).get("scenario_errors_log")
        if errors_log_str:
            errors_log = Path(errors_log_str)
            if errors_log.exists() and errors_log.stat().st_size > 0:
                failures.append(
                    CheckFailure(
                        "Scenario errors (OpenMATB)",
                        [str(errors_log)] + _read_text_lines(errors_log, max_lines=50),
                    )
                )
                # Skip marker/timing checks for this scenario; it did not run to completion.
                continue

        # Manifest checks (aligned to current runner contract)
        if manifest.get("seq_id") != args.seq_id:
            failures.append(CheckFailure("Manifest seq_id mismatch", [str(manifest_path)]))
        if "dry_run" in manifest and manifest.get("dry_run") is not False:
            failures.append(CheckFailure("Manifest dry_run is not false", [str(manifest_path)]))
        if manifest.get("unattended") is not False:
            failures.append(CheckFailure("Manifest unattended not false", [str(manifest_path)]))
        if manifest.get("scenario_name") != scenario_stem:
            failures.append(CheckFailure("Manifest scenario_name mismatch", [str(manifest_path)]))

        output_dir = Path(str(manifest.get("output_dir", "")))
        if output_dir and repo_root in output_dir.parents:
            failures.append(CheckFailure("Output dir is inside repo", [str(output_dir)]))

        csv_path_str = manifest.get("paths", {}).get("session_csv")
        if not csv_path_str:
            failures.append(CheckFailure("Manifest missing paths.session_csv", [str(manifest_path)]))
            continue
        csv_path = Path(csv_path_str)
        if not csv_path.exists():
            failures.append(CheckFailure("CSV missing", [str(csv_path)]))
            continue

        csv_events = vps._parse_csv_events(csv_path)
        if not csv_events:
            failures.append(CheckFailure("CSV has no events", [str(csv_path)]))
            continue

        mono = _check_monotonic_scenario_time(csv_events, csv_path)
        if mono:
            failures.append(mono)

        marker_fail = _check_markers_match_scenario(
            scenario_path=scenario_path,
            csv_path=csv_path,
            participant=args.participant,
            session=args.session,
            seq_id=args.seq_id,
        )
        if marker_fail:
            failures.append(marker_fail)

        lines, seg_failures = _check_dynamic_segments(
            scenario_path=scenario_path,
            csv_path=csv_path,
            tolerance_sec=args.duration_tolerance_seconds,
        )
        print(f"\n--- {scenario_filename} ---")
        print(f"manifest: {manifest_path}")
        print(f"csv: {csv_path}")
        for line in lines[:200]:
            print(line)
        failures.extend(seg_failures)

    if failures:
        print("\nFAILED")
        for f in failures:
            print(f"- {f.label}")
            for d in f.details[:20]:
                print(f"  - {d}")
        return 1

    print("\nPASS: all checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
