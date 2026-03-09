"""Pilot scenario + CSV verifier utilities.

This module exists to support `src/python/verification/verify_pilot.py`.

Scope (Pilot 1):
- Parse repo-managed scenario text files under `scenarios/`.
- Parse OpenMATB session CSV rows needed for marker/segment checks.
- Provide small, deterministic utilities used by dynamic verification.

Non-goals:
- Do not enforce any non-canonical protocol contracts.
- Do not introduce new protocol semantics; prefer observability.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


TOKEN_PID = "${OPENMATB_PARTICIPANT}"
TOKEN_SID = "${OPENMATB_SESSION}"
TOKEN_SEQ = "${OPENMATB_SEQ_ID}"

ALLOWED_LEVELS = {"LOW", "MODERATE", "HIGH"}


# Pilot 1 block duration (practice/calibration scenarios).
BLOCK_DURATION_SEC = 300.0

# Guardrail: distributed demand events should not fire at the very start/end of the block.
EVENT_EDGE_BUFFER_SEC = 5.0

# Vendor examples recover pump failures by setting state back to `off` after 10s.
RESMAN_PUMP_FAILURE_DURATION_SEC = 10.0


@dataclass(frozen=True)
class Event:
    line_no: int
    time_sec: float
    plugin: str
    command: list[str]


@dataclass(frozen=True)
class Segment:
    label: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CsvEvent:
    row_index: int
    scenario_time: float
    module: str
    address: str
    value: str


@dataclass(frozen=True)
class CheckEvent:
    label: str


def _parse_time_seconds(raw: str) -> Optional[float]:
    raw = raw.strip()
    if not raw or ":" not in raw:
        return None
    try:
        h, m, s = raw.split(":")
        return float(int(h) * 3600 + int(m) * 60 + int(s))
    except Exception:
        return None


def _parse_events(scenario_path: Path) -> list[Event]:
    events: list[Event] = []
    for line_no, raw in enumerate(scenario_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3:
            continue
        t = _parse_time_seconds(parts[0])
        if t is None:
            continue
        plugin = parts[1].strip().lower()
        cmd = [p.strip() for p in parts[2:] if p.strip()]
        events.append(Event(line_no=line_no, time_sec=float(t), plugin=plugin, command=cmd))
    return events


def _find_marker_events(events: Iterable[Event]) -> list[Event]:
    out: list[Event] = []
    for ev in events:
        if ev.plugin != "labstreaminglayer":
            continue
        if not ev.command:
            continue
        if ev.command[0].strip().lower() != "marker":
            continue
        out.append(ev)
    return out


def _marker_name(marker_payload: str) -> str:
    """Return marker name without STUDY/V0/ prefix and without payload metadata."""

    raw = str(marker_payload or "").strip()
    if not raw:
        return ""
    raw = raw.split("|", 1)[0].strip()
    if raw.startswith("STUDY/V0/"):
        raw = raw[len("STUDY/V0/") :]
    return raw


def _parse_csv_events(csv_path: Path) -> list[CsvEvent]:
    events: list[CsvEvent] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            t_raw = str(row.get("scenario_time") or "").strip()
            try:
                t = float(t_raw)
            except Exception:
                continue
            module = str(row.get("module") or "").strip().lower()
            address = str(row.get("address") or "").strip().lower()
            value = "" if row.get("value") is None else str(row.get("value"))
            events.append(CsvEvent(row_index=row_index, scenario_time=float(t), module=module, address=address, value=value))
    return events


def _extract_markers_from_csv(csv_path: Path) -> list[str]:
    markers: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip().lower()
            module = (row.get("module") or "").strip().lower()
            address = (row.get("address") or "").strip().lower()
            if row_type != "event" or module != "labstreaminglayer" or address != "marker":
                continue
            value = str(row.get("value") or "")
            if value:
                markers.append(value)
    return markers


def _label_for_marker_base(base: str) -> str:
    if base.startswith("TRAINING/"):
        # TRAINING/T1 -> T1
        return base.split("/", 1)[1]
    if base.startswith("calibration/"):
        # calibration/LOW -> calibration_LOW
        return base.replace("/", "_")
    if base.startswith("TLX/"):
        # TLX/calibration_LOW -> TLX_calibration_LOW
        return base.replace("/", "_")
    return base.replace("/", "_")


def _collect_csv_segments(csv_events: list[CsvEvent]) -> dict[str, Segment]:
    """Collect first start/end pair per marker base into segments.

    Returns a mapping keyed by segment label (e.g., T1, calibration_LOW).
    """

    starts: dict[str, tuple[float, int]] = {}
    ends: dict[str, tuple[float, int]] = {}

    for ev in csv_events:
        if ev.module != "labstreaminglayer" or ev.address != "marker":
            continue
        name = _marker_name(ev.value)
        if not name:
            continue
        if name.endswith("/START"):
            base = name[: -len("/START")]
            if base not in starts:
                starts[base] = (ev.scenario_time, ev.row_index)
        elif name.endswith("/END"):
            base = name[: -len("/END")]
            if base not in ends:
                ends[base] = (ev.scenario_time, ev.row_index)

    segments: dict[str, Segment] = {}
    for base, (start_t, start_row) in starts.items():
        end = ends.get(base)
        if end is None:
            continue
        end_t, end_row = end
        if end_t <= start_t:
            continue
        label = _label_for_marker_base(base)
        segments[label] = Segment(
            label=label,
            start_sec=float(start_t),
            end_sec=float(end_t),
            duration_sec=float(end_t - start_t),
            start_line=int(start_row),
            end_line=int(end_row),
        )
    return segments


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    out = [fmt_row(headers), "-+-".join("-" * w for w in widths)]
    out.extend(fmt_row([str(c) for c in r]) for r in rows)
    return out


def _dynamic_duration_check(
    scenario_name: str,
    intended: dict[str, Segment],
    observed: dict[str, Segment],
    tolerance_sec: float,
    csv_path: Path,
) -> list[CheckEvent]:
    headers = ["Segment", "Intended(s)", "Observed(s)", "Delta(s)", "PASS"]
    rows: list[list[str]] = []

    for label, seg in sorted(intended.items()):
        if label not in observed:
            continue
        obs = observed[label]
        delta = obs.duration_sec - seg.duration_sec
        ok = abs(delta) <= tolerance_sec
        rows.append(
            [
                label,
                f"{seg.duration_sec:.2f}",
                f"{obs.duration_sec:.2f}",
                f"{delta:+.2f}",
                "PASS" if ok else "FAIL",
            ]
        )

    lines = [f"Duration check: {scenario_name} ({csv_path.name})"]
    if rows:
        lines.extend(_format_table(headers, rows))
    else:
        lines.append("(no comparable segments)")
    return [CheckEvent(label=l) for l in lines]


def _match_expected_times(
    expected: list[float],
    observed: list[float],
    tolerance_sec: float,
) -> tuple[list[float], list[float]]:
    missing: list[float] = []
    extra = list(observed)

    for e in expected:
        match_idx = None
        for i, o in enumerate(extra):
            if abs(o - e) <= tolerance_sec:
                match_idx = i
                break
        if match_idx is None:
            missing.append(e)
        else:
            extra.pop(match_idx)

    return missing, extra


def _dynamic_comm_schedule_check(
    scenario_name: str,
    expected_comm_times: list[float],
    csv_events: list[CsvEvent],
    tolerance_sec: float,
    csv_path: Path,
) -> list[CheckEvent]:
    observed_comm_times = sorted(
        [e.scenario_time for e in csv_events if e.module == "communications" and e.address == "radioprompt"]
    )
    missing, extra = _match_expected_times(expected_comm_times, observed_comm_times, tolerance_sec)
    ok = not missing and not extra

    lines = [
        f"Comms schedule check: {scenario_name} ({csv_path.name}) :: PASS" if ok else f"Comms schedule check: {scenario_name} ({csv_path.name}) :: FAIL",
        f"expected_n={len(expected_comm_times)} observed_n={len(observed_comm_times)} tol={tolerance_sec}s",
    ]
    if missing:
        lines.append(f"missing (first 10): {missing[:10]}")
    if extra:
        lines.append(f"extra (first 10): {extra[:10]}")

    return [CheckEvent(label=l) for l in lines]


def _static_check_scenario_file(path: Path) -> list[str]:
    """Return list of problems; empty means pass."""

    problems: list[str] = []
    try:
        events = _parse_events(path)
    except Exception as exc:
        return [f"Unable to parse {path}: {exc}"]

    if not events:
        # Intro scenario intentionally has no markers; still must be parseable.
        return []

    # Basic sanity: non-decreasing timestamps
    prev = None
    for ev in events:
        if prev is not None and ev.time_sec < prev:
            problems.append(f"Non-monotonic time at line {ev.line_no}: {ev.time_sec} < {prev}")
            break
        prev = ev.time_sec

    # Marker token sanity: ensure markers (if present) still contain templated tokens.
    for ev in _find_marker_events(events):
        if not ev.command or len(ev.command) < 2:
            continue
        payload = ev.command[-1]
        if "STUDY/V0/" not in payload:
            problems.append(f"Marker missing STUDY/V0 prefix at line {ev.line_no}")
        if TOKEN_PID not in payload or TOKEN_SID not in payload or TOKEN_SEQ not in payload:
            problems.append(f"Marker missing pid/sid/seq tokens at line {ev.line_no}")

    # ResMan pump failure recovery sanity: every failure should be followed by off at +10s.
    # This is a solvability constraint (task must recover), not a protocol semantics constraint.
    duration_ms = int(round(RESMAN_PUMP_FAILURE_DURATION_SEC * 1000.0))
    resman_state_events: list[tuple[int, int, str, str]] = []
    for ev in events:
        if ev.plugin != "resman" or len(ev.command) < 2:
            continue
        address = ev.command[0].strip().lower()
        value = ev.command[1].strip().lower()
        if not address.startswith("pump-") or not address.endswith("-state"):
            continue
        if value not in {"failure", "off"}:
            continue
        t_ms = int(round(ev.time_sec * 1000.0))
        resman_state_events.append((t_ms, ev.line_no, address, value))

    if resman_state_events:
        present = {(t_ms, address, value) for (t_ms, _line, address, value) in resman_state_events}
        for t_ms, line_no, address, value in resman_state_events:
            if value != "failure":
                continue
            expected_t_ms = t_ms + duration_ms
            if (expected_t_ms, address, "off") not in present:
                expected_sec = expected_t_ms / 1000.0
                problems.append(
                    f"ResMan pump failure not recovered at +{int(RESMAN_PUMP_FAILURE_DURATION_SEC)}s: "
                    f"{address} failure at line {line_no} (t={t_ms/1000.0:.3f}) missing off at t={expected_sec:.3f}"
                )

    # Edge-buffer sanity for distributed *demand* events.
    # Note: we intentionally do NOT apply this to task start/stop/marker events.
    min_t = EVENT_EDGE_BUFFER_SEC
    max_t = BLOCK_DURATION_SEC - EVENT_EDGE_BUFFER_SEC

    def is_demand_event(ev: Event) -> bool:
        if not ev.command:
            return False
        cmd0 = ev.command[0].strip().lower()
        if ev.plugin == "sysmon" and cmd0.endswith("-failure"):
            return True
        if ev.plugin == "communications" and cmd0 == "radioprompt":
            return True
        if ev.plugin == "resman" and cmd0.startswith("pump-") and cmd0.endswith("-state"):
            # Includes both failure + off recovery events.
            return True
        return False

    for ev in events:
        if not is_demand_event(ev):
            continue
        if ev.time_sec < min_t or ev.time_sec > max_t:
            problems.append(
                f"Demand event within edge buffer at line {ev.line_no}: t={ev.time_sec:.3f}s (allowed {min_t:.1f}..{max_t:.1f})"
            )

    return problems


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scenario_dir = repo_root / "experiment" / "scenarios"
    if not scenario_dir.exists():
        print(f"ERROR: scenarios dir not found: {scenario_dir}")
        return 2

    # Verify only the scenarios the runner may use (Pilot 1).
    required = [
        "pilot_practice_intro.txt",
        "pilot_practice_low.txt",
        "pilot_practice_moderate.txt",
        "pilot_practice_high.txt",
        "pilot_calibration_low.txt",
        "pilot_calibration_moderate.txt",
        "pilot_calibration_high.txt",
    ]

    failures: list[str] = []
    for name in required:
        path = scenario_dir / name
        if not path.exists():
            failures.append(f"Missing scenario: {name}")
            continue
        problems = _static_check_scenario_file(path)
        for p in problems:
            failures.append(f"{name}: {p}")

    if failures:
        print("FAILED")
        for f in failures:
            print(f"- {f}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
