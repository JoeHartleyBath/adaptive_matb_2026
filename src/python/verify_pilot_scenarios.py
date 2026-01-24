"""Verify pilot scenario artifacts and (optionally) unattended run outputs.

Usage:
    python src/python/verify_pilot_scenarios.py
    python src/python/verify_pilot_scenarios.py --output-root C:/data/adaptive_matb --participant P001 --session S001 --seq-id SEQ1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

TOKEN_PID = "${OPENMATB_PARTICIPANT}"
TOKEN_SID = "${OPENMATB_SESSION}"
TOKEN_SEQ = "${OPENMATB_SEQ_ID}"

REPO_SPEC_FILES = [
    "docs/pilot/PILOT_STUDY_SPEC_V0.md",
    "docs/contracts/training_scenario_contract_v0.md",
    "docs/pilot/PILOT_BUILD_PLAN_V0.md",
    "docs/DATA_MANAGEMENT.md",
]

SEQ_LEVELS = {
    "SEQ1": ["LOW", "MODERATE", "HIGH"],
    "SEQ2": ["MODERATE", "HIGH", "LOW"],
    "SEQ3": ["HIGH", "LOW", "MODERATE"],
}

LEVEL_COUNTS = {
    "LOW": {"sysmon": 5, "communications": 5, "resman": 5},
    "MODERATE": {"sysmon": 15, "communications": 15, "resman": 10},
    "HIGH": {"sysmon": 30, "communications": 30, "resman": 30},
}

LEVEL_RATES = {
    "LOW": {"sysmon": 1, "communications": 1, "resman": 1},
    "MODERATE": {"sysmon": 3, "communications": 3, "resman": 2},
    "HIGH": {"sysmon": 6, "communications": 6, "resman": 6},
}

ALLOWED_LEVELS = {"LOW", "MODERATE", "HIGH"}

NON_BLOCKING_PLUGINS = {
    "sysmon",
    "track",
    "communications",
    "resman",
    "scheduling",
    "labstreaminglayer",
}

BLOCK_DURATION_SEC = 300

FRENCH_STOPWORDS = {
    "le",
    "la",
    "les",
    "des",
    "un",
    "une",
    "et",
    "mais",
    "ou",
    "où",
    "avec",
    "pour",
    "vous",
    "être",
    "pas",
    "ne",
    "ce",
    "cette",
    "ces",
    "dans",
    "sur",
}

MODULE_NAME_PATTERN = re.compile(r"\b(SYSMON|RESMAN|TRACK|COMM)\b", flags=re.IGNORECASE)


@dataclass
class Evidence:
    label: str
    link: Optional[str] = None


@dataclass
class RequirementResult:
    req_id: str
    section: str
    category: str
    requirement: str
    status: str
    evidence: list[Evidence] = field(default_factory=list)
    rationale: Optional[str] = None


@dataclass
class Event:
    line_no: int
    time_sec: int
    plugin: str
    command: list[str]
    raw: str


@dataclass
class Block:
    name: str
    kind: str
    level: str
    start_sec: int
    end_sec: int
    start_line: int
    end_line: int


@dataclass
class Segment:
    label: str
    start_sec: float
    end_sec: float
    duration_sec: float
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class CsvEvent:
    row_index: int
    scenario_time: float
    module: str
    address: str
    value: str
    raw: dict[str, str]


def marker(name: str) -> str:
    return f"STUDY/V0/{name}|pid={TOKEN_PID}|sid={TOKEN_SID}|seq={TOKEN_SEQ}"


def _expected_markers(seq_id: str) -> list[str]:
    retained = SEQ_LEVELS[seq_id]
    markers = [
        marker("SESSION_START"),
        marker("TRAINING/T1/START"),
        marker("TRAINING/T1/END"),
        marker("TRAINING/T2/START"),
        marker("TRAINING/T2/END"),
        marker("TRAINING/T3/START"),
        marker("TRAINING/T3/END"),
    ]
    for idx, level in enumerate(retained, start=1):
        markers.append(marker(f"RETAINED/B{idx}/{level}/START"))
        markers.append(marker(f"RETAINED/B{idx}/{level}/END"))
        markers.append(marker(f"TLX/B{idx}/START"))
        markers.append(marker(f"TLX/B{idx}/END"))
    markers.append(marker("SESSION_END"))
    return markers


def _expected_dry_run_markers(
    participant: str | None = None,
    session: str | None = None,
    seq_id: str | None = None,
) -> list[str]:
    markers = [
        marker("SESSION_START"),
        marker("TRAINING/T1/START"),
        marker("TRAINING/T1/END"),
        marker("RETAINED/B1/LOW/START"),
        marker("RETAINED/B1/LOW/END"),
        marker("TLX/B1/START"),
        marker("TLX/B1/END"),
        marker("SESSION_END"),
    ]
    if participant and session and seq_id:
        markers = [
            m.replace(TOKEN_PID, participant)
            .replace(TOKEN_SID, session)
            .replace(TOKEN_SEQ, seq_id)
            for m in markers
        ]
    return markers


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _time_to_sec(time_str: str) -> int:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _parse_events(path: Path) -> list[Event]:
    events: list[Event] = []
    for idx, line in enumerate(_read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(";")
        if len(parts) < 3:
            continue
        time_str = parts[0]
        plugin = parts[1]
        command = parts[2:]
        try:
            time_sec = _time_to_sec(time_str)
        except ValueError:
            continue
        events.append(Event(idx, time_sec, plugin, command, stripped))
    return events


def _parse_csv_events(csv_path: Path) -> list[CsvEvent]:
    events: list[CsvEvent] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=2):
            scenario_time = row.get("scenario_time")
            if scenario_time is None or scenario_time == "":
                continue
            try:
                time_val = float(scenario_time)
            except ValueError:
                continue
            events.append(
                CsvEvent(
                    row_index=idx,
                    scenario_time=time_val,
                    module=row.get("module", ""),
                    address=row.get("address", ""),
                    value=row.get("value", ""),
                    raw=row,
                )
            )
    return events


def _rel_link(repo_root: Path, path: Path, line_no: Optional[int] = None) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    rel_str = rel.as_posix()
    if line_no:
        return f"[{rel_str}]({rel_str}#L{line_no})"
    return f"[{rel_str}]({rel_str})"


def _add_pass(results: list[RequirementResult], req_id: str, section: str, category: str, requirement: str, evidence: list[Evidence]) -> None:
    results.append(
        RequirementResult(
            req_id=req_id,
            section=section,
            category=category,
            requirement=requirement,
            status="PASS",
            evidence=evidence,
        )
    )


def _add_fail(results: list[RequirementResult], req_id: str, section: str, category: str, requirement: str, evidence: list[Evidence]) -> None:
    results.append(
        RequirementResult(
            req_id=req_id,
            section=section,
            category=category,
            requirement=requirement,
            status="FAIL",
            evidence=evidence,
        )
    )


def _add_nmv(results: list[RequirementResult], req_id: str, section: str, category: str, requirement: str, rationale: str) -> None:
    results.append(
        RequirementResult(
            req_id=req_id,
            section=section,
            category=category,
            requirement=requirement,
            status="NOT MACHINE-VERIFIABLE",
            rationale=rationale,
        )
    )


def _find_manifest(output_root: Path, participant: str, session: str) -> Optional[Path]:
    sessions_dir = output_root / "openmatb" / participant / session / "sessions"
    if not sessions_dir.exists():
        return None
    manifests = sorted(sessions_dir.glob("**/*.manifest.json"))
    if not manifests:
        return None
    return manifests[-1]


def _find_latest_csv(manifest_path: Path) -> Optional[Path]:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    csv_path = manifest.get("paths", {}).get("session_csv")
    if not csv_path:
        return None
    path = Path(csv_path)
    return path if path.exists() else None


def _find_marker_events(events: Iterable[Event]) -> list[Event]:
    return [e for e in events if e.plugin == "labstreaminglayer" and len(e.command) >= 2 and e.command[0] == "marker"]


def _marker_name(payload: str) -> Optional[str]:
    if not payload.startswith("STUDY/V0/"):
        return None
    if "|" not in payload:
        return None
    return payload.split("|", 1)[0].replace("STUDY/V0/", "")


def _block_from_marker(name: str) -> Optional[tuple[str, str, str]]:
    if name.startswith("TRAINING/") and name.endswith("/START"):
        block = name.split("/")[1]
        level = {"T1": "LOW", "T2": "MODERATE", "T3": "HIGH"}.get(block)
        if level:
            return (block, "training", level)
    if name.startswith("RETAINED/") and name.endswith("/START"):
        parts = name.split("/")
        if len(parts) == 4:
            block = parts[1]
            level = parts[2]
            if level in ALLOWED_LEVELS:
                return (block, "retained", level)
    return None


def _collect_blocks(events: list[Event]) -> list[Block]:
    markers = _find_marker_events(events)
    start_markers: dict[str, Event] = {}
    end_markers: dict[str, Event] = {}
    blocks: list[Block] = []

    for marker_event in markers:
        payload = marker_event.command[-1]
        name = _marker_name(payload)
        if not name:
            continue
        if name.endswith("/START"):
            block_info = _block_from_marker(name)
            if block_info:
                key = f"{block_info[0]}_{block_info[1]}"
                start_markers[key] = marker_event
        if name.endswith("/END"):
            if name.startswith("TRAINING/"):
                block = name.split("/")[1]
                key = f"{block}_training"
                end_markers[key] = marker_event
            if name.startswith("RETAINED/"):
                block = name.split("/")[1]
                key = f"{block}_retained"
                end_markers[key] = marker_event

    for key, start_event in start_markers.items():
        if key not in end_markers:
            continue
        end_event = end_markers[key]
        block_name, kind = key.split("_")
        if kind == "training":
            level = {"T1": "LOW", "T2": "MODERATE", "T3": "HIGH"}[block_name]
        else:
            payload = start_event.command[-1]
            marker_name = _marker_name(payload)
            level = marker_name.split("/")[2]
        blocks.append(
            Block(
                name=block_name,
                kind=kind,
                level=level,
                start_sec=start_event.time_sec,
                end_sec=end_event.time_sec,
                start_line=start_event.line_no,
                end_line=end_event.line_no,
            )
        )
    return sorted(blocks, key=lambda b: b.start_sec)


def _collect_csv_segments(csv_events: list[CsvEvent]) -> dict[str, Segment]:
    marker_events = [e for e in csv_events if e.module == "labstreaminglayer" and e.address == "marker"]
    markers: dict[str, list[CsvEvent]] = {}
    for ev in marker_events:
        name = _marker_name(ev.value)
        if name:
            markers.setdefault(name, []).append(ev)

    segments: dict[str, Segment] = {}

    def add_segment(label: str, start_marker: str, end_marker: str) -> None:
        if start_marker not in markers or end_marker not in markers:
            return
        start = markers[start_marker][0]
        end = markers[end_marker][0]
        segments[label] = Segment(
            label=label,
            start_sec=start.scenario_time,
            end_sec=end.scenario_time,
            duration_sec=end.scenario_time - start.scenario_time,
            start_line=start.row_index,
            end_line=end.row_index,
        )

    add_segment("T1", "TRAINING/T1/START", "TRAINING/T1/END")
    add_segment("T2", "TRAINING/T2/START", "TRAINING/T2/END")
    add_segment("T3", "TRAINING/T3/START", "TRAINING/T3/END")

    for idx in (1, 2, 3):
        add_segment(f"B{idx}", f"RETAINED/B{idx}/LOW/START", f"RETAINED/B{idx}/LOW/END")
        add_segment(f"B{idx}", f"RETAINED/B{idx}/MODERATE/START", f"RETAINED/B{idx}/MODERATE/END")
        add_segment(f"B{idx}", f"RETAINED/B{idx}/HIGH/START", f"RETAINED/B{idx}/HIGH/END")
        add_segment(f"TLX/B{idx}", f"TLX/B{idx}/START", f"TLX/B{idx}/END")

    return segments


def _extract_markers_from_csv(csv_path: Path) -> list[str]:
    markers: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") != "event":
                continue
            if row.get("module") != "labstreaminglayer":
                continue
            if row.get("address") != "marker":
                continue
            value = row.get("value")
            if value:
                markers.append(value)
    return markers


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, col in enumerate(row):
            widths[idx] = max(widths[idx], len(col))
    line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    sep = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    output = [line, sep]
    for row in rows:
        output.append(" | ".join(col.ljust(widths[idx]) for idx, col in enumerate(row)))
    return output


def _check_spec_files_exist(repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for rel in REPO_SPEC_FILES:
        path = repo_root / rel
        if not path.exists():
            evidence.append(Evidence(f"Missing spec file {rel}"))
        else:
            evidence.append(Evidence(f"Found spec file {rel}", _rel_link(repo_root, path, 1)))
    return evidence


def _training_hashes(scenario_paths: list[Path], repo_root: Path) -> tuple[dict[str, str], list[Evidence]]:
    hashes: dict[str, str] = {}
    evidence: list[Evidence] = []
    for path in scenario_paths:
        lines = _read_text(path).splitlines()
        start = end = None
        for idx, line in enumerate(lines):
            if "TRAINING/T1/START" in line:
                start = idx
            if "TRAINING/T3/END" in line:
                end = idx
        if start is None or end is None:
            evidence.append(Evidence("Training markers missing", _rel_link(repo_root, path, 1)))
            continue
        segment = "\n".join(lines[start : end + 1]).encode("utf-8")
        digest = hashlib.sha256(segment).hexdigest()
        hashes[path.name] = digest
        evidence.append(Evidence(f"Training hash {path.name}: {digest}", _rel_link(repo_root, path, start + 1)))
    return hashes, evidence


def _check_scenario_grammar(path: Path, repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for idx, line in enumerate(_read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(";")
        if len(parts) not in (3, 4):
            evidence.append(Evidence(f"Invalid field count {len(parts)}", _rel_link(repo_root, path, idx)))
            continue
        time_str = parts[0]
        if not re.match(r"^\d+:\d{2}:\d{2}$", time_str):
            evidence.append(Evidence(f"Invalid time format {time_str}", _rel_link(repo_root, path, idx)))
    return evidence


def _check_marker_payloads(path: Path, events: list[Event], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for event in _find_marker_events(events):
        payload = event.command[-1]
        if ";" in payload:
            evidence.append(Evidence("Marker payload contains semicolon", _rel_link(repo_root, path, event.line_no)))
        if not payload.startswith("STUDY/V0/"):
            evidence.append(Evidence("Marker payload missing STUDY/V0 prefix", _rel_link(repo_root, path, event.line_no)))
        if f"pid={TOKEN_PID}" not in payload or f"sid={TOKEN_SID}" not in payload or f"seq={TOKEN_SEQ}" not in payload:
            evidence.append(Evidence("Marker payload missing pid/sid/seq tokens", _rel_link(repo_root, path, event.line_no)))
    return evidence


def _check_plugin_start_stop(path: Path, events: list[Event], repo_root: Path) -> tuple[list[Evidence], list[Evidence]]:
    failures: list[Evidence] = []
    evidence: list[Evidence] = []
    plugins = {e.plugin for e in events}
    for plugin in sorted(plugins):
        start_lines = [e.line_no for e in events if e.plugin == plugin and e.command and e.command[0] == "start"]
        stop_lines = [e.line_no for e in events if e.plugin == plugin and e.command and e.command[0] == "stop"]
        if not start_lines:
            failures.append(Evidence(f"Missing start for plugin {plugin}", _rel_link(repo_root, path, 1)))
            continue
        if plugin in NON_BLOCKING_PLUGINS and not stop_lines:
            failures.append(Evidence(f"Missing stop for plugin {plugin}", _rel_link(repo_root, path, start_lines[0])))
        if start_lines:
            evidence.append(Evidence(f"Plugin {plugin} start", _rel_link(repo_root, path, start_lines[0])))
        if stop_lines:
            evidence.append(Evidence(f"Plugin {plugin} stop", _rel_link(repo_root, path, stop_lines[0])))
    return failures, evidence


def _expected_offsets(count: int) -> list[int]:
    return [15 + ((i + 1) * 275) // (count + 1) for i in range(count)]


def _assign_offsets(base_offsets: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    assigned: dict[str, list[int]] = {"sysmon": [], "communications": [], "resman": []}
    used: set[int] = set()
    last_comm: Optional[int] = None

    for task, offset, _idx in base_offsets:
        current = offset
        while True:
            if current < 15 or current > 289:
                raise ValueError("Offset out of bounds")
            if current in used:
                current += 1
                continue
            if task == "communications" and last_comm is not None and current - last_comm < 8:
                current += 1
                continue
            break
        assigned[task].append(current)
        used.add(current)
        if task == "communications":
            last_comm = current
    return assigned


def _ordered_base_offsets(counts: dict[str, int]) -> list[tuple[str, int, int]]:
    base: list[tuple[str, int, int]] = []
    for task in ("sysmon", "communications", "resman"):
        offsets = _expected_offsets(counts[task])
        for idx, offset in enumerate(offsets):
            base.append((task, offset, idx))
    priority = {"sysmon": 0, "communications": 1, "resman": 2}
    base.sort(key=lambda x: (x[1], priority[x[0]], x[2]))
    return base


def _block_task_events(events: list[Event], block: Block) -> dict[str, list[Event]]:
    task_events: dict[str, list[Event]] = {"sysmon": [], "communications": [], "resman": []}
    for event in events:
        if event.time_sec < block.start_sec or event.time_sec >= block.end_sec:
            continue
        if event.plugin == "sysmon" and len(event.command) == 2 and "failure" in event.command[0]:
            task_events["sysmon"].append(event)
        if event.plugin == "communications" and event.command and event.command[0] == "radioprompt":
            task_events["communications"].append(event)
        if event.plugin == "resman" and len(event.command) == 2 and "pump" in event.command[0] and event.command[1] == "failure":
            task_events["resman"].append(event)
    return task_events


def _check_block_schedule(path: Path, events: list[Event], blocks: list[Block], repo_root: Path) -> tuple[list[Evidence], list[Evidence]]:
    failures: list[Evidence] = []
    evidence: list[Evidence] = []
    for block in blocks:
        counts = LEVEL_COUNTS[block.level]
        task_events = _block_task_events(events, block)

        # Count checks
        for task, expected_count in counts.items():
            actual_count = len(task_events[task])
            if actual_count != expected_count:
                failures.append(
                    Evidence(
                        f"{block.name} {block.level} {task} count {actual_count} != {expected_count}",
                        _rel_link(repo_root, path, block.start_line),
                    )
                )

        # Offset checks
        offsets: dict[str, list[int]] = {}
        for task, task_list in task_events.items():
            offsets[task] = [e.time_sec - block.start_sec for e in task_list]
            for offset, event in zip(offsets[task], task_list):
                if offset < 15 or offset > 289:
                    failures.append(
                        Evidence(
                            f"{block.name} {task} offset {offset} out of bounds",
                            _rel_link(repo_root, path, event.line_no),
                        )
                    )

        # Overlap rule
        used: dict[int, list[str]] = {}
        for task, offset_list in offsets.items():
            for offset in offset_list:
                used.setdefault(offset, []).append(task)
        overlaps = {k: v for k, v in used.items() if len(set(v)) > 1}
        evidence.append(Evidence(f"{block.name} collision count: {len(overlaps)}", _rel_link(repo_root, path, block.start_line)))
        for offset, tasks in overlaps.items():
            failures.append(
                Evidence(
                    f"{block.name} overlap at {offset}s: {sorted(set(tasks))}",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

        # Communications spacing and target/distractor assignment
        comm_events = task_events["communications"]
        comm_offsets = [e.time_sec - block.start_sec for e in comm_events]
        min_spacing = None
        for prev, curr in zip(comm_offsets, comm_offsets[1:]):
            spacing = curr - prev
            min_spacing = spacing if min_spacing is None else min(min_spacing, spacing)
            if spacing < 8:
                failures.append(
                    Evidence(
                        f"{block.name} communications spacing {prev}->{curr}",
                        _rel_link(repo_root, path, comm_events[0].line_no) if comm_events else _rel_link(repo_root, path, block.start_line),
                    )
                )
        if min_spacing is not None:
            evidence.append(Evidence(f"{block.name} min comm spacing: {min_spacing}s", _rel_link(repo_root, path, block.start_line)))

        for idx, event in enumerate(comm_events, start=1):
            expected = "other" if idx % 5 == 0 else "own"
            actual = event.command[1] if len(event.command) > 1 else ""
            if actual != expected:
                failures.append(
                    Evidence(
                        f"{block.name} comm #{idx} expected {expected} got {actual}",
                        _rel_link(repo_root, path, event.line_no),
                    )
                )

        # Deterministic schedule check
        base_offsets = _ordered_base_offsets(counts)
        try:
            assigned = _assign_offsets(base_offsets)
        except ValueError as exc:
            failures.append(
                Evidence(
                    f"{block.name} deterministic assignment failed: {exc}",
                    _rel_link(repo_root, path, block.start_line),
                )
            )
            continue

        for task in ("sysmon", "communications", "resman"):
            expected_offsets = assigned[task]
            actual_offsets = offsets[task]
            if expected_offsets != actual_offsets:
                failures.append(
                    Evidence(
                        f"{block.name} {task} offsets mismatch expected {expected_offsets[:5]}... got {actual_offsets[:5]}...",
                        _rel_link(repo_root, path, block.start_line),
                    )
                )

    return failures, evidence


def _segment_table_from_markers(path: Path, events: list[Event], repo_root: Path) -> list[Evidence]:
    segments: dict[str, Segment] = {}
    marker_events = _find_marker_events(events)
    marker_by_name: dict[str, Event] = {}
    for event in marker_events:
        name = _marker_name(event.command[-1])
        if name:
            marker_by_name[name] = event

    def add_segment(label: str, start_marker: str, end_marker: str) -> None:
        if start_marker not in marker_by_name or end_marker not in marker_by_name:
            return
        start = marker_by_name[start_marker]
        end = marker_by_name[end_marker]
        segments[label] = Segment(
            label=label,
            start_sec=start.time_sec,
            end_sec=end.time_sec,
            duration_sec=end.time_sec - start.time_sec,
            start_line=start.line_no,
            end_line=end.line_no,
        )

    add_segment("T1", "TRAINING/T1/START", "TRAINING/T1/END")
    add_segment("T2", "TRAINING/T2/START", "TRAINING/T2/END")
    add_segment("T3", "TRAINING/T3/START", "TRAINING/T3/END")

    for idx in (1, 2, 3):
        add_segment(f"B{idx}", f"RETAINED/B{idx}/LOW/START", f"RETAINED/B{idx}/LOW/END")
        add_segment(f"B{idx}", f"RETAINED/B{idx}/MODERATE/START", f"RETAINED/B{idx}/MODERATE/END")
        add_segment(f"B{idx}", f"RETAINED/B{idx}/HIGH/START", f"RETAINED/B{idx}/HIGH/END")
        add_segment(f"TLX/B{idx}", f"TLX/B{idx}/START", f"TLX/B{idx}/END")

    rows: list[list[str]] = []
    for key in sorted(segments.keys()):
        seg = segments[key]
        rows.append([seg.label, f"{seg.start_sec}", f"{seg.end_sec}", f"{seg.duration_sec}"])

    table = _format_table(["Segment", "Start", "End", "Duration"], rows)
    evidence = [Evidence(f"Static timing table for {path.name}", _rel_link(repo_root, path, 1))]
    evidence.extend(Evidence(line) for line in table)
    return evidence


def _check_session_structure(path: Path, blocks: list[Block], events: list[Event], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []

    training = [b for b in blocks if b.kind == "training"]
    retained = [b for b in blocks if b.kind == "retained"]

    # Training order and durations
    expected_training = ["T1", "T2", "T3"]
    if [b.name for b in training] != expected_training:
        evidence.append(Evidence("Training order mismatch", _rel_link(repo_root, path, training[0].start_line if training else 1)))
    for block in training:
        duration = block.end_sec - block.start_sec
        if duration != BLOCK_DURATION_SEC:
            evidence.append(
                Evidence(
                    f"{block.name} duration {duration}s != 300s",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

    # Breaks between training blocks
    for prev, nxt in zip(training, training[1:]):
        gap = nxt.start_sec - prev.end_sec
        if gap != 60:
            evidence.append(
                Evidence(
                    f"Training break {prev.name}->{nxt.name} {gap}s != 60s",
                    _rel_link(repo_root, path, nxt.start_line),
                )
            )

    # Retained blocks duration and breaks
    for block in retained:
        duration = block.end_sec - block.start_sec
        if duration != BLOCK_DURATION_SEC:
            evidence.append(
                Evidence(
                    f"{block.name} duration {duration}s != 300s",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

    # TLX immediately after retained block end
    marker_events = _find_marker_events(events)
    tlx_start = {e.time_sec: e for e in marker_events if _marker_name(e.command[-1]) and _marker_name(e.command[-1]).startswith("TLX/") and _marker_name(e.command[-1]).endswith("/START")}
    tlx_end = {e.time_sec: e for e in marker_events if _marker_name(e.command[-1]) and _marker_name(e.command[-1]).startswith("TLX/") and _marker_name(e.command[-1]).endswith("/END")}
    for block in retained:
        if block.end_sec not in tlx_start:
            evidence.append(
                Evidence(
                    f"{block.name} missing TLX START at block end",
                    _rel_link(repo_root, path, block.end_line),
                )
            )
        if block.end_sec not in tlx_end:
            evidence.append(
                Evidence(
                    f"{block.name} missing TLX END at block end",
                    _rel_link(repo_root, path, block.end_line),
                )
            )

    # Breaks between retained blocks (1:00 after TLX)
    retained_sorted = sorted(retained, key=lambda b: b.start_sec)
    for prev, nxt in zip(retained_sorted, retained_sorted[1:]):
        gap = nxt.start_sec - prev.end_sec
        if gap != 60:
            evidence.append(
                Evidence(
                    f"Retained break {prev.name}->{nxt.name} {gap}s != 60s",
                    _rel_link(repo_root, path, nxt.start_line),
                )
            )

    return evidence


def _check_retained_order(path: Path, events: list[Event], seq_id: str, repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    marker_events = _find_marker_events(events)
    retained = []
    for event in marker_events:
        name = _marker_name(event.command[-1])
        if name and name.startswith("RETAINED/") and name.endswith("/START"):
            retained.append(name.split("/")[2])
    expected = SEQ_LEVELS[seq_id]
    evidence.append(Evidence(f"Retained order extracted: {retained}", _rel_link(repo_root, path, marker_events[0].line_no if marker_events else 1)))
    if retained != expected:
        evidence.append(Evidence(f"Retained order mismatch expected {expected}", _rel_link(repo_root, path, 1)))
    return evidence


def _check_markers_against_spec(path: Path, events: list[Event], seq_id: str, repo_root: Path) -> tuple[list[Evidence], list[Evidence]]:
    failures: list[Evidence] = []
    evidence: list[Evidence] = []
    expected = _expected_markers(seq_id)
    marker_events = _find_marker_events(events)
    payloads = {e.command[-1]: e for e in marker_events}
    for marker_text in expected:
        if marker_text not in payloads:
            failures.append(Evidence(f"Missing marker {marker_text}", _rel_link(repo_root, path, 1)))
        else:
            event = payloads[marker_text]
            evidence.append(Evidence(f"Marker present {marker_text}", _rel_link(repo_root, path, event.line_no)))

    for event in marker_events:
        payload = event.command[-1]
        name = _marker_name(payload)
        if not name or not name.startswith("RETAINED/"):
            continue
        parts = name.split("/")
        if len(parts) == 4:
            level = parts[2]
            if level not in ALLOWED_LEVELS:
                failures.append(
                    Evidence(
                        f"Invalid retained level {level}",
                        _rel_link(repo_root, path, event.line_no),
                    )
                )

    return failures, evidence


def _check_instruction_assets(path: Path, instructions_dir: Path, repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    content = _read_text(path)
    instruction_files = []
    for idx, line in enumerate(content.splitlines(), start=1):
        if "instructions;filename" in line:
            parts = line.split(";")
            if len(parts) >= 4:
                instruction_files.append((parts[3], idx))

    if not instruction_files:
        evidence.append(Evidence("No instruction assets referenced", _rel_link(repo_root, path, 1)))
        return evidence

    for filename, line_no in instruction_files:
        inst_path = instructions_dir / filename
        if not inst_path.exists():
            evidence.append(Evidence(f"Instruction file missing: {filename}", _rel_link(repo_root, path, line_no)))
            continue
        text = _read_text(inst_path)
        lowered = re.findall(r"[\w']+", text.lower())
        french_hits = [w for w in lowered if w in FRENCH_STOPWORDS]
        module_hits = MODULE_NAME_PATTERN.findall(text)
        excerpt = " ".join(text.split())[:200].encode("utf-8")
        excerpt_hash = hashlib.sha256(excerpt).hexdigest()
        evidence.append(Evidence(f"Instruction file {filename} hash {excerpt_hash}", _rel_link(repo_root, inst_path, 1)))
        if len(french_hits) >= 3:
            evidence.append(Evidence(f"French stopword hits: {sorted(set(french_hits))}", _rel_link(repo_root, inst_path, 1)))
        if module_hits:
            evidence.append(Evidence(f"Module name hits: {sorted(set(module_hits))}", _rel_link(repo_root, inst_path, 1)))

    return evidence


def _check_training_identical(scenario_paths: list[Path], repo_root: Path) -> tuple[bool, list[Evidence]]:
    hashes, evidence = _training_hashes(scenario_paths, repo_root)
    if len(set(hashes.values())) != 1:
        evidence.append(Evidence("Training segment hashes do not match", _rel_link(repo_root, scenario_paths[0], 1)))
        return False, evidence
    return True, evidence


def _check_wrapper_mapping(wrapper_path: Path, filenames: list[str], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    if not wrapper_path.exists():
        evidence.append(Evidence("Wrapper not found", _rel_link(repo_root, wrapper_path)))
        return evidence
    wrapper_text = _read_text(wrapper_path)
    for name in filenames:
        if name not in wrapper_text:
            evidence.append(Evidence(f"Wrapper missing scenario mapping {name}", _rel_link(repo_root, wrapper_path, 1)))
        else:
            line_no = next((idx for idx, line in enumerate(wrapper_text.splitlines(), start=1) if name in line), 1)
            evidence.append(Evidence(f"Wrapper mapping {name}", _rel_link(repo_root, wrapper_path, line_no)))
    return evidence


def _check_manifest_requirements(
    manifest_path: Path,
    repo_root: Path,
    expected_scenario: str,
    seq_id: str,
    dry_run: bool,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("unattended") is not True:
        evidence.append(Evidence("Manifest unattended != true", manifest_path.as_posix()))
    if manifest.get("dry_run") is not dry_run:
        evidence.append(Evidence("Manifest dry_run mismatch", manifest_path.as_posix()))
    if manifest.get("seq_id") != seq_id:
        evidence.append(Evidence("Manifest seq_id mismatch", manifest_path.as_posix()))
    if manifest.get("scenario_name") != expected_scenario:
        evidence.append(Evidence("Manifest scenario_name mismatch", manifest_path.as_posix()))
    if not manifest.get("repo_commit") or not manifest.get("submodule_commit"):
        evidence.append(Evidence("Manifest missing commit metadata", manifest_path.as_posix()))

    output_dir = Path(manifest.get("output_dir", ""))
    if output_dir and repo_root in output_dir.parents:
        evidence.append(Evidence("Output directory inside repo", manifest_path.as_posix()))

    if not evidence:
        evidence.append(Evidence(f"Manifest OK {manifest_path}", manifest_path.as_posix()))
    return evidence


def _check_dry_run_csv_markers(csv_path: Path, participant: str, session: str, seq_id: str) -> list[Evidence]:
    evidence: list[Evidence] = []
    markers = _extract_markers_from_csv(csv_path)
    expected = _expected_dry_run_markers(participant, session, seq_id)
    for marker_text in expected:
        if marker_text not in markers:
            evidence.append(Evidence(f"Missing CSV marker {marker_text}", csv_path.as_posix()))

    if not evidence:
        evidence.append(Evidence(f"CSV markers OK {csv_path}", csv_path.as_posix()))
    return evidence


def _scan_repo_for_output_logs(repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    vendor_sessions = repo_root / "src" / "python" / "vendor" / "openmatb" / "sessions"
    for path in repo_root.rglob("*.csv"):
        if vendor_sessions in path.parents:
            continue
        if "sessions" in path.parts:
            evidence.append(Evidence(f"Output CSV found in repo: {path}", _rel_link(repo_root, path, 1)))
    for path in repo_root.rglob("*.manifest.json"):
        if vendor_sessions in path.parents:
            continue
        if "sessions" in path.parts:
            evidence.append(Evidence(f"Manifest found in repo: {path}", _rel_link(repo_root, path, 1)))
    return evidence


def _dynamic_duration_check(
    label: str,
    intended: dict[str, Segment],
    observed: dict[str, Segment],
    tolerance_sec: float,
    csv_path: Path,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    headers = ["Segment", "Intended(s)", "Observed(s)", "Delta(s)", "WithinTol"]
    rows: list[list[str]] = []
    for key, intended_seg in intended.items():
        if key not in observed:
            rows.append([key, f"{intended_seg.duration_sec:.2f}", "MISSING", "", "FAIL"])
            continue
        observed_seg = observed[key]
        delta = observed_seg.duration_sec - intended_seg.duration_sec
        ok = abs(delta) <= tolerance_sec
        rows.append(
            [
                key,
                f"{intended_seg.duration_sec:.2f}",
                f"{observed_seg.duration_sec:.2f}",
                f"{delta:.2f}",
                "PASS" if ok else "FAIL",
            ]
        )
    evidence.append(Evidence(f"{label} timing comparison (tol={tolerance_sec}s) :: {csv_path}"))
    evidence.extend(Evidence(line) for line in _format_table(headers, rows))
    return evidence


def _dynamic_event_counts(
    label: str,
    csv_events: list[CsvEvent],
    block_segments: dict[str, Segment],
    expected_counts: dict[str, dict[str, int]],
    csv_path: Path,
) -> list[Evidence]:
    headers = ["Block", "Level", "Sysmon", "Comms", "Resman", "Total", "Events/min", "Expected", "PASS"]
    rows: list[list[str]] = []

    def is_sysmon(ev: CsvEvent) -> bool:
        return ev.module == "sysmon" and "failure" in ev.address and ev.value.lower() in {"true", "1"}

    def is_comm(ev: CsvEvent) -> bool:
        return ev.module == "communications" and ev.address == "radioprompt"

    def is_resman(ev: CsvEvent) -> bool:
        return ev.module == "resman" and "pump" in ev.address and ev.value == "failure"

    for block_name, segment in block_segments.items():
        if not block_name.startswith("B"):
            continue
        level = "UNKNOWN"
        for level_key in ALLOWED_LEVELS:
            if f"/{level_key}/" in block_name or level_key in block_name:
                level = level_key
        expected = expected_counts.get(level, {})
        relevant = [e for e in csv_events if segment.start_sec <= e.scenario_time < segment.end_sec]
        sysmon_count = sum(1 for e in relevant if is_sysmon(e))
        comm_count = sum(1 for e in relevant if is_comm(e))
        resman_count = sum(1 for e in relevant if is_resman(e))
        total = sysmon_count + comm_count + resman_count
        events_per_min = total / ((segment.duration_sec or 1) / 60)
        expected_total = sum(expected.values())
        pass_block = (
            expected
            and sysmon_count == expected.get("sysmon", -1)
            and comm_count == expected.get("communications", -1)
            and resman_count == expected.get("resman", -1)
        )
        rows.append(
            [
                block_name,
                level,
                str(sysmon_count),
                str(comm_count),
                str(resman_count),
                str(total),
                f"{events_per_min:.2f}",
                str(expected_total),
                "PASS" if pass_block else "FAIL",
            ]
        )

    evidence = [Evidence(f"{label} event counts from CSV :: {csv_path}")]
    evidence.extend(Evidence(line) for line in _format_table(headers, rows))
    return evidence


def _match_expected_times(expected: list[float], observed: list[float], tolerance: float) -> tuple[list[float], list[float]]:
    missing: list[float] = []
    extra = observed.copy()
    for exp in expected:
        match = None
        for obs in extra:
            if abs(obs - exp) <= tolerance:
                match = obs
                break
        if match is None:
            missing.append(exp)
        else:
            extra.remove(match)
    return missing, extra


def _dynamic_comm_schedule_check(
    label: str,
    expected_times: list[float],
    csv_events: list[CsvEvent],
    tolerance: float,
    csv_path: Path,
) -> list[Evidence]:
    observed_times = sorted([e.scenario_time for e in csv_events if e.module == "communications" and e.address == "radioprompt"])
    missing, extra = _match_expected_times(expected_times, observed_times, tolerance)
    evidence: list[Evidence] = [Evidence(f"{label} comm schedule check tol={tolerance}s :: {csv_path}")]
    evidence.append(Evidence(f"Expected comm count: {len(expected_times)}"))
    evidence.append(Evidence(f"Observed comm count: {len(observed_times)}"))
    if missing:
        evidence.append(Evidence(f"Missing comm times: {missing[:10]}"))
    if extra:
        evidence.append(Evidence(f"Extra comm times: {extra[:10]}"))
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pilot scenario artifacts and optional unattended outputs.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--seq-id", choices=("SEQ1", "SEQ2", "SEQ3"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-scenario", default="pilot_dry_run_v0.txt")
    parser.add_argument("--csv-dry-run", type=Path, default=None)
    parser.add_argument("--csv-full-run", type=Path, default=None)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=0.5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scenario_dir = repo_root / "src" / "python" / "vendor" / "openmatb" / "includes" / "scenarios"
    instructions_dir = repo_root / "src" / "python" / "vendor" / "openmatb" / "includes" / "instructions"
    scenario_paths = [
        scenario_dir / "pilot_seq1.txt",
        scenario_dir / "pilot_seq2.txt",
        scenario_dir / "pilot_seq3.txt",
    ]
    dry_run_path = scenario_dir / args.dry_run_scenario
    wrapper_path = repo_root / "src" / "python" / "run_openmatb.py"

    results: list[RequirementResult] = []

    # STATIC CHECKS
    spec_evidence = _check_spec_files_exist(repo_root)
    if any("Missing" in e.label for e in spec_evidence):
        _add_fail(results, "A1", "STATIC CHECKS", "Artifact presence & naming", "Spec/contract files exist", spec_evidence)
    else:
        _add_pass(results, "A1", "STATIC CHECKS", "Artifact presence & naming", "Spec/contract files exist", spec_evidence)

    scenario_evidence: list[Evidence] = []
    missing = False
    for path in scenario_paths + [dry_run_path]:
        if not path.exists():
            scenario_evidence.append(Evidence(f"Missing scenario {path}"))
            missing = True
        else:
            scenario_evidence.append(Evidence(f"Found scenario {path.name}", _rel_link(repo_root, path, 1)))
    if missing:
        _add_fail(results, "A2", "STATIC CHECKS", "Artifact presence & naming", "Scenario files exist", scenario_evidence)
    else:
        _add_pass(results, "A2", "STATIC CHECKS", "Artifact presence & naming", "Scenario files exist", scenario_evidence)

    wrapper_evidence = _check_wrapper_mapping(wrapper_path, [p.name for p in scenario_paths + [dry_run_path]], repo_root)
    if any("missing" in e.label.lower() for e in wrapper_evidence):
        _add_fail(results, "A3", "STATIC CHECKS", "Artifact presence & naming", "Wrapper references all scenarios", wrapper_evidence)
    else:
        _add_pass(results, "A3", "STATIC CHECKS", "Artifact presence & naming", "Wrapper references all scenarios", wrapper_evidence)

    training_ok, training_evidence = _check_training_identical(scenario_paths, repo_root)
    if not training_ok:
        _add_fail(results, "A4", "STATIC CHECKS", "Artifact presence & naming", "Training segments identical across SEQ1-SEQ3", training_evidence)
    else:
        _add_pass(results, "A4", "STATIC CHECKS", "Artifact presence & naming", "Training segments identical across SEQ1-SEQ3", training_evidence)

    grammar_failures: list[Evidence] = []
    payload_failures: list[Evidence] = []
    parser_failures: list[Evidence] = []
    parser_evidence: list[Evidence] = []
    for path in scenario_paths + [dry_run_path]:
        events = _parse_events(path)
        grammar_failures.extend(_check_scenario_grammar(path, repo_root))
        payload_failures.extend(_check_marker_payloads(path, events, repo_root))
        failures, evidence = _check_plugin_start_stop(path, events, repo_root)
        parser_failures.extend(failures)
        parser_evidence.extend(evidence)

    if grammar_failures:
        _add_fail(results, "B1", "STATIC CHECKS", "Scenario grammar & parser compliance", "OpenMATB line format and time format", grammar_failures)
    else:
        _add_pass(results, "B1", "STATIC CHECKS", "Scenario grammar & parser compliance", "OpenMATB line format and time format", [Evidence("All lines valid", _rel_link(repo_root, scenario_paths[0], 1))])

    if parser_failures:
        _add_fail(results, "B2", "STATIC CHECKS", "Scenario grammar & parser compliance", "Plugins have start/stop as required", parser_failures)
    else:
        _add_pass(results, "B2", "STATIC CHECKS", "Scenario grammar & parser compliance", "Plugins have start/stop as required", parser_evidence or [Evidence("All required plugin start/stop found", _rel_link(repo_root, scenario_paths[0], 1))])

    if payload_failures:
        _add_fail(results, "D1", "STATIC CHECKS", "Marker presence, naming, ordering", "Marker payload format and token presence", payload_failures)
    else:
        _add_pass(results, "D1", "STATIC CHECKS", "Marker presence, naming, ordering", "Marker payload format and token presence", [Evidence("All marker payloads valid", _rel_link(repo_root, scenario_paths[0], 1))])

    structure_failures: list[Evidence] = []
    static_timing_evidence: list[Evidence] = []
    retained_order_failures: list[Evidence] = []
    for seq_id, path in zip(("SEQ1", "SEQ2", "SEQ3"), scenario_paths):
        events = _parse_events(path)
        blocks = _collect_blocks(events)
        structure_failures.extend(_check_session_structure(path, blocks, events, repo_root))
        static_timing_evidence.extend(_segment_table_from_markers(path, events, repo_root))
        retained_order_failures.extend(_check_retained_order(path, events, seq_id, repo_root))

    if structure_failures:
        _add_fail(results, "C1", "STATIC CHECKS", "Session structure semantics", "Training/retained duration and break timing", structure_failures)
    else:
        _add_pass(results, "C1", "STATIC CHECKS", "Session structure semantics", "Training/retained duration and break timing", static_timing_evidence)

    if any("mismatch" in e.label.lower() for e in retained_order_failures):
        _add_fail(results, "C2", "STATIC CHECKS", "Session structure semantics", "Retained order matches SEQ mapping", retained_order_failures)
    else:
        _add_pass(results, "C2", "STATIC CHECKS", "Session structure semantics", "Retained order matches SEQ mapping", retained_order_failures)

    marker_failures: list[Evidence] = []
    marker_evidence: list[Evidence] = []
    for seq_id, path in zip(("SEQ1", "SEQ2", "SEQ3"), scenario_paths):
        events = _parse_events(path)
        failures, evidence = _check_markers_against_spec(path, events, seq_id, repo_root)
        marker_failures.extend(failures)
        marker_evidence.extend(evidence)
    if marker_failures:
        _add_fail(results, "D2", "STATIC CHECKS", "Marker presence, naming, ordering", "Marker list matches spec", marker_failures)
    else:
        _add_pass(results, "D2", "STATIC CHECKS", "Marker presence, naming, ordering", "Marker list matches spec", marker_evidence)

    scheduling_failures: list[Evidence] = []
    scheduling_evidence: list[Evidence] = []
    for path in scenario_paths:
        events = _parse_events(path)
        blocks = _collect_blocks(events)
        failures, evidence = _check_block_schedule(path, events, blocks, repo_root)
        scheduling_failures.extend(failures)
        scheduling_evidence.extend(evidence)
    if scheduling_failures:
        _add_fail(results, "E1", "STATIC CHECKS", "Determinism & scheduling rules", "Per-block deterministic schedule + overlap rules", scheduling_failures)
    else:
        _add_pass(results, "E1", "STATIC CHECKS", "Determinism & scheduling rules", "Per-block deterministic schedule + overlap rules", scheduling_evidence)

    instruction_failures: list[Evidence] = []
    for path in scenario_paths:
        instruction_failures.extend(_check_instruction_assets(path, instructions_dir, repo_root))
    if any("missing" in e.label.lower() or "stopword" in e.label.lower() or "module name" in e.label.lower() for e in instruction_failures):
        _add_fail(results, "F1", "STATIC CHECKS", "Instruction asset policy", "Instruction assets exist and are English", instruction_failures)
    else:
        _add_pass(results, "F1", "STATIC CHECKS", "Instruction asset policy", "Instruction assets exist and are English", instruction_failures)

    log_scan = _scan_repo_for_output_logs(repo_root)
    if log_scan:
        _add_fail(results, "G2", "STATIC CHECKS", "Data-management boundaries", "No output logs stored in repo", log_scan)
    else:
        _add_pass(results, "G2", "STATIC CHECKS", "Data-management boundaries", "No output logs stored in repo", [Evidence("No session logs found in repo", _rel_link(repo_root, repo_root, 1))])

    # DYNAMIC CHECKS (dry-run)
    if args.csv_dry_run and args.csv_dry_run.exists():
        dry_csv = args.csv_dry_run
    elif args.output_root and args.participant and args.session:
        manifest_path = _find_manifest(args.output_root, args.participant, args.session)
        dry_csv = _find_latest_csv(manifest_path) if manifest_path else None
    else:
        dry_csv = None

    if dry_csv and dry_csv.exists() and args.participant and args.session and args.seq_id:
        csv_marker_evidence = _check_dry_run_csv_markers(dry_csv, args.participant, args.session, args.seq_id)
        if any("Missing" in e.label for e in csv_marker_evidence):
            _add_fail(results, "D3", "DYNAMIC CHECKS (dry-run CSV)", "Marker presence", "CSV markers present", csv_marker_evidence)
        else:
            _add_pass(results, "D3", "DYNAMIC CHECKS (dry-run CSV)", "Marker presence", "CSV markers present", csv_marker_evidence)
    else:
        _add_nmv(
            results,
            "D3",
            "DYNAMIC CHECKS (dry-run CSV)",
            "Marker presence",
            "CSV markers present",
            "Requires --csv-dry-run or --output-root/--participant/--session/--seq-id.",
        )

    # DYNAMIC CHECKS (full-run)
    if args.csv_full_run and args.csv_full_run.exists() and args.seq_id:
        full_csv = args.csv_full_run
        full_events = _parse_csv_events(full_csv)
        intended_events = _parse_events(scenario_paths[int(args.seq_id[-1]) - 1])
        observed_segments = _collect_csv_segments(full_events)

        intended_seg_map: dict[str, Segment] = {}
        marker_events = _find_marker_events(intended_events)
        marker_by_name = { _marker_name(e.command[-1]): e for e in marker_events if _marker_name(e.command[-1]) }
        for label in ("T1", "T2", "T3"):
            start = marker_by_name.get(f"TRAINING/{label}/START")
            end = marker_by_name.get(f"TRAINING/{label}/END")
            if start and end:
                intended_seg_map[label] = Segment(label, start.time_sec, end.time_sec, end.time_sec - start.time_sec)
        for idx in (1, 2, 3):
            for level in ALLOWED_LEVELS:
                start = marker_by_name.get(f"RETAINED/B{idx}/{level}/START")
                end = marker_by_name.get(f"RETAINED/B{idx}/{level}/END")
                if start and end:
                    intended_seg_map[f"B{idx}"] = Segment(f"B{idx}", start.time_sec, end.time_sec, end.time_sec - start.time_sec)
            start = marker_by_name.get(f"TLX/B{idx}/START")
            end = marker_by_name.get(f"TLX/B{idx}/END")
            if start and end:
                intended_seg_map[f"TLX/B{idx}"] = Segment(f"TLX/B{idx}", start.time_sec, end.time_sec, end.time_sec - start.time_sec)

        duration_evidence = _dynamic_duration_check("Full-run", intended_seg_map, observed_segments, args.duration_tolerance_seconds, full_csv)
        if any("FAIL" in e.label for e in duration_evidence):
            _add_fail(results, "C3", "DYNAMIC CHECKS (full-run CSV)", "Session structure timing", "Observed durations match intended", duration_evidence)
        else:
            _add_pass(results, "C3", "DYNAMIC CHECKS (full-run CSV)", "Session structure timing", "Observed durations match intended", duration_evidence)

        counts_evidence = _dynamic_event_counts("Full-run", full_events, observed_segments, LEVEL_COUNTS, full_csv)
        if any("FAIL" in e.label for e in counts_evidence):
            _add_fail(results, "C4", "DYNAMIC CHECKS (full-run CSV)", "Workload/event-rate compliance", "Observed counts match expected", counts_evidence)
        else:
            _add_pass(results, "C4", "DYNAMIC CHECKS (full-run CSV)", "Workload/event-rate compliance", "Observed counts match expected", counts_evidence)

        expected_comm_times: list[float] = []
        for ev in intended_events:
            if ev.plugin == "communications" and ev.command and ev.command[0] == "radioprompt":
                expected_comm_times.append(float(ev.time_sec))
        comm_evidence = _dynamic_comm_schedule_check("Full-run", expected_comm_times, full_events, args.duration_tolerance_seconds, full_csv)
        if any("Missing" in e.label or "Extra" in e.label for e in comm_evidence):
            _add_fail(results, "E2", "DYNAMIC CHECKS (full-run CSV)", "Communications schedule", "Observed comm prompts match intended", comm_evidence)
        else:
            _add_pass(results, "E2", "DYNAMIC CHECKS (full-run CSV)", "Communications schedule", "Observed comm prompts match intended", comm_evidence)
    else:
        _add_nmv(
            results,
            "C3",
            "DYNAMIC CHECKS (full-run CSV)",
            "Session structure timing",
            "Observed durations match intended",
            "Requires --csv-full-run and --seq-id for full-run log validation.",
        )
        _add_nmv(
            results,
            "C4",
            "DYNAMIC CHECKS (full-run CSV)",
            "Workload/event-rate compliance",
            "Observed counts match expected",
            "Requires --csv-full-run and --seq-id for full-run log validation.",
        )
        _add_nmv(
            results,
            "E2",
            "DYNAMIC CHECKS (full-run CSV)",
            "Communications schedule",
            "Observed comm prompts match intended",
            "Requires --csv-full-run and --seq-id for full-run log validation.",
        )

    # NOT MACHINE-VERIFIABLE
    _add_nmv(
        results,
        "N1",
        "NOT MACHINE-VERIFIABLE",
        "Session structure semantics",
        "Participant ID entry popup blocks start until valid ID submitted",
        "Requires UI interaction and human input; not verifiable statically.",
    )
    _add_nmv(
        results,
        "N2",
        "NOT MACHINE-VERIFIABLE",
        "Marker transport requirements",
        "LSL marker presence when LSL is chosen timebase",
        "Suggested harness: run dry-run with pylsl listener to capture marker stream.",
    )
    _add_nmv(
        results,
        "N3",
        "NOT MACHINE-VERIFIABLE",
        "Marker set",
        "ABORT marker emitted only on early termination",
        "Suggested harness: force abort in unattended mode and assert ABORT marker in CSV.",
    )
    _add_nmv(
        results,
        "N4",
        "NOT MACHINE-VERIFIABLE",
        "Session structure semantics",
        "NASA-TLX self-paced completion with all sliders interacted",
        "Requires participant interaction; not statically verifiable.",
    )
    _add_nmv(
        results,
        "N5",
        "NOT MACHINE-VERIFIABLE",
        "Provisional alignment tolerance",
        "EEG/OpenMATB alignment tolerance ≤ 20 ms",
        "Requires EEG acquisition stack and timing measurements.",
    )
    _add_nmv(
        results,
        "N6",
        "NOT MACHINE-VERIFIABLE",
        "Pause policy",
        "No pause during retained blocks; abort on interruption",
        "Operational procedure requirement; not detectable statically.",
    )

    print("VERIFICATION REPORT")
    print("-------------------")
    for section in ["STATIC CHECKS", "DYNAMIC CHECKS (dry-run CSV)", "DYNAMIC CHECKS (full-run CSV)", "NOT MACHINE-VERIFIABLE"]:
        print(section)
        print("-")
        for result in results:
            if result.section != section:
                continue
            print(f"[{result.status}] {result.req_id} {result.category} :: {result.requirement}")
            for ev in result.evidence:
                if ev.link:
                    print(f"  - {ev.label} :: {ev.link}")
                else:
                    print(f"  - {ev.label}")
            if result.rationale:
                print(f"  - RATIONALE: {result.rationale}")
        print("")

    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

LEVEL_RATES = {
    "LOW": {"sysmon": 1, "communications": 1, "resman": 1},
    "MODERATE": {"sysmon": 3, "communications": 3, "resman": 2},
    "HIGH": {"sysmon": 6, "communications": 6, "resman": 6},
}

ALLOWED_LEVELS = {"LOW", "MODERATE", "HIGH"}

NON_BLOCKING_PLUGINS = {
    "sysmon",
    "track",
    "communications",
    "resman",
    "scheduling",
    "labstreaminglayer",
}

BLOCK_DURATION_SEC = 300


@dataclass
class Evidence:
    label: str
    link: Optional[str] = None


@dataclass
class RequirementResult:
    req_id: str
    category: str
    requirement: str
    status: str
    evidence: list[Evidence] = field(default_factory=list)
    rationale: Optional[str] = None


@dataclass
class Event:
    line_no: int
    time_sec: int
    plugin: str
    command: list[str]
    raw: str


@dataclass
class Block:
    name: str
    kind: str
    level: str
    start_sec: int
    end_sec: int
    start_line: int
    end_line: int


def marker(name: str) -> str:
    return f"STUDY/V0/{name}|pid={TOKEN_PID}|sid={TOKEN_SID}|seq={TOKEN_SEQ}"


def _expected_markers(seq_id: str) -> list[str]:
    retained = SEQ_LEVELS[seq_id]
    markers = [
        marker("SESSION_START"),
        marker("TRAINING/T1/START"),
        marker("TRAINING/T1/END"),
        marker("TRAINING/T2/START"),
        marker("TRAINING/T2/END"),
        marker("TRAINING/T3/START"),
        marker("TRAINING/T3/END"),
    ]
    for idx, level in enumerate(retained, start=1):
        markers.append(marker(f"RETAINED/B{idx}/{level}/START"))
        markers.append(marker(f"RETAINED/B{idx}/{level}/END"))
        markers.append(marker(f"TLX/B{idx}/START"))
        markers.append(marker(f"TLX/B{idx}/END"))
    markers.append(marker("SESSION_END"))
    return markers


def _expected_dry_run_markers(
    participant: str | None = None,
    session: str | None = None,
    seq_id: str | None = None,
) -> list[str]:
    markers = [
        marker("SESSION_START"),
        marker("TRAINING/T1/START"),
        marker("TRAINING/T1/END"),
        marker("RETAINED/B1/LOW/START"),
        marker("RETAINED/B1/LOW/END"),
        marker("TLX/B1/START"),
        marker("TLX/B1/END"),
        marker("SESSION_END"),
    ]
    if participant and session and seq_id:
        markers = [
            m.replace("${OPENMATB_PARTICIPANT}", participant)
            .replace("${OPENMATB_SESSION}", session)
            .replace("${OPENMATB_SEQ_ID}", seq_id)
            for m in markers
        ]
    return markers


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _time_to_sec(time_str: str) -> int:
    h, m, s = time_str.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _parse_events(path: Path) -> list[Event]:
    events: list[Event] = []
    for idx, line in enumerate(_read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(";")
        if len(parts) < 3:
            continue
        time_str = parts[0]
        plugin = parts[1]
        command = parts[2:]
        try:
            time_sec = _time_to_sec(time_str)
        except ValueError:
            continue
        events.append(Event(idx, time_sec, plugin, command, stripped))
    return events


def _rel_link(repo_root: Path, path: Path, line_no: Optional[int] = None) -> str:
    rel = path.resolve().relative_to(repo_root.resolve())
    rel_str = rel.as_posix()
    if line_no:
        return f"[{rel_str}]({rel_str}#L{line_no})"
    return f"[{rel_str}]({rel_str})"


def _add_pass(results: list[RequirementResult], req_id: str, category: str, requirement: str, evidence: list[Evidence]) -> None:
    results.append(RequirementResult(req_id=req_id, category=category, requirement=requirement, status="PASS", evidence=evidence))


def _add_fail(results: list[RequirementResult], req_id: str, category: str, requirement: str, evidence: list[Evidence]) -> None:
    results.append(RequirementResult(req_id=req_id, category=category, requirement=requirement, status="FAIL", evidence=evidence))


def _add_nmv(results: list[RequirementResult], req_id: str, category: str, requirement: str, rationale: str) -> None:
    results.append(RequirementResult(req_id=req_id, category=category, requirement=requirement, status="NOT MACHINE-VERIFIABLE", rationale=rationale))


def _find_manifest(output_root: Path, participant: str, session: str) -> Optional[Path]:
    sessions_dir = output_root / "openmatb" / participant / session / "sessions"
    if not sessions_dir.exists():
        return None
    manifests = sorted(sessions_dir.glob("**/*.manifest.json"))
    if not manifests:
        return None
    return manifests[-1]


def _find_latest_csv(manifest_path: Path) -> Optional[Path]:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    csv_path = manifest.get("paths", {}).get("session_csv")
    if not csv_path:
        return None
    path = Path(csv_path)
    return path if path.exists() else None


def _find_marker_events(events: Iterable[Event]) -> list[Event]:
    return [e for e in events if e.plugin == "labstreaminglayer" and len(e.command) >= 2 and e.command[0] == "marker"]


def _marker_name(payload: str) -> Optional[str]:
    if not payload.startswith("STUDY/V0/"):
        return None
    if "|" not in payload:
        return None
    return payload.split("|", 1)[0].replace("STUDY/V0/", "")


def _block_from_marker(name: str) -> Optional[tuple[str, str, str]]:
    if name.startswith("TRAINING/") and name.endswith("/START"):
        block = name.split("/")[1]
        level = {"T1": "LOW", "T2": "MODERATE", "T3": "HIGH"}.get(block)
        if level:
            return (block, "training", level)
    if name.startswith("RETAINED/") and name.endswith("/START"):
        parts = name.split("/")
        if len(parts) == 4:
            block = parts[1]
            level = parts[2]
            if level in ALLOWED_LEVELS:
                return (block, "retained", level)
    return None


def _collect_blocks(events: list[Event]) -> list[Block]:
    markers = _find_marker_events(events)
    start_markers: dict[str, Event] = {}
    end_markers: dict[str, Event] = {}
    blocks: list[Block] = []

    for marker_event in markers:
        payload = marker_event.command[-1]
        name = _marker_name(payload)
        if not name:
            continue
        if name.endswith("/START"):
            block_info = _block_from_marker(name)
            if block_info:
                key = f"{block_info[0]}_{block_info[1]}"
                start_markers[key] = marker_event
        if name.endswith("/END"):
            if name.startswith("TRAINING/"):
                block = name.split("/")[1]
                key = f"{block}_training"
                end_markers[key] = marker_event
            if name.startswith("RETAINED/"):
                block = name.split("/")[1]
                key = f"{block}_retained"
                end_markers[key] = marker_event

    for key, start_event in start_markers.items():
        if key not in end_markers:
            continue
        end_event = end_markers[key]
        block_name, kind = key.split("_")
        if kind == "training":
            level = {"T1": "LOW", "T2": "MODERATE", "T3": "HIGH"}[block_name]
        else:
            payload = start_event.command[-1]
            marker_name = _marker_name(payload)
            level = marker_name.split("/")[2]
        blocks.append(
            Block(
                name=block_name,
                kind=kind,
                level=level,
                start_sec=start_event.time_sec,
                end_sec=end_event.time_sec,
                start_line=start_event.line_no,
                end_line=end_event.line_no,
            )
        )
    return sorted(blocks, key=lambda b: b.start_sec)


def _check_scenario_grammar(path: Path, events: list[Event], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for idx, line in enumerate(_read_text(path).splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(";")
        if len(parts) not in (3, 4):
            evidence.append(Evidence(f"Invalid field count {len(parts)}", _rel_link(repo_root, path, idx)))
            continue
        time_str = parts[0]
        if not re.match(r"^\d+:\d{2}:\d{2}$", time_str):
            evidence.append(Evidence(f"Invalid time format {time_str}", _rel_link(repo_root, path, idx)))
    return evidence


def _check_marker_payloads(path: Path, events: list[Event], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for event in _find_marker_events(events):
        payload = event.command[-1]
        if ";" in payload:
            evidence.append(Evidence("Marker payload contains semicolon", _rel_link(repo_root, path, event.line_no)))
        if not payload.startswith("STUDY/V0/"):
            evidence.append(Evidence("Marker payload missing STUDY/V0 prefix", _rel_link(repo_root, path, event.line_no)))
        if f"pid={TOKEN_PID}" not in payload or f"sid={TOKEN_SID}" not in payload or f"seq={TOKEN_SEQ}" not in payload:
            evidence.append(Evidence("Marker payload missing pid/sid/seq tokens", _rel_link(repo_root, path, event.line_no)))
    return evidence


def _check_plugin_start_stop(path: Path, events: list[Event], repo_root: Path) -> tuple[list[Evidence], list[Evidence]]:
    failures: list[Evidence] = []
    evidence: list[Evidence] = []
    plugins = {e.plugin for e in events}
    for plugin in sorted(plugins):
        start_lines = [e.line_no for e in events if e.plugin == plugin and e.command and e.command[0] == "start"]
        stop_lines = [e.line_no for e in events if e.plugin == plugin and e.command and e.command[0] == "stop"]
        if not start_lines:
            failures.append(Evidence(f"Missing start for plugin {plugin}", _rel_link(repo_root, path, 1)))
            continue
        if plugin in NON_BLOCKING_PLUGINS and not stop_lines:
            failures.append(Evidence(f"Missing stop for plugin {plugin}", _rel_link(repo_root, path, start_lines[0])))
        if start_lines:
            evidence.append(Evidence(f"Plugin {plugin} start", _rel_link(repo_root, path, start_lines[0])))
        if stop_lines:
            evidence.append(Evidence(f"Plugin {plugin} stop", _rel_link(repo_root, path, stop_lines[0])))
    return failures, evidence


def _extract_markers_from_csv(csv_path: Path) -> list[str]:
    markers: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") != "event":
                continue
            if row.get("module") != "labstreaminglayer":
                continue
            if row.get("address") != "marker":
                continue
            value = row.get("value")
            if value:
                markers.append(value)
    return markers


def _expected_offsets(count: int) -> list[int]:
    return [15 + ((i + 1) * 275) // (count + 1) for i in range(count)]


def _assign_offsets(base_offsets: list[tuple[str, int, int]]) -> dict[str, list[int]]:
    assigned: dict[str, list[int]] = {"sysmon": [], "communications": [], "resman": []}
    used: set[int] = set()
    last_comm: Optional[int] = None

    for task, offset, _idx in base_offsets:
        current = offset
        while True:
            if current < 15 or current > 289:
                raise ValueError("Offset out of bounds")
            if current in used:
                current += 1
                continue
            if task == "communications" and last_comm is not None and current - last_comm < 8:
                current += 1
                continue
            break
        assigned[task].append(current)
        used.add(current)
        if task == "communications":
            last_comm = current
    return assigned


def _ordered_base_offsets(counts: dict[str, int]) -> list[tuple[str, int, int]]:
    base: list[tuple[str, int, int]] = []
    for task in ("sysmon", "communications", "resman"):
        offsets = _expected_offsets(counts[task])
        for idx, offset in enumerate(offsets):
            base.append((task, offset, idx))
    priority = {"sysmon": 0, "communications": 1, "resman": 2}
    base.sort(key=lambda x: (x[1], priority[x[0]], x[2]))
    return base


def _block_task_events(events: list[Event], block: Block) -> dict[str, list[Event]]:
    task_events: dict[str, list[Event]] = {"sysmon": [], "communications": [], "resman": []}
    for event in events:
        if event.time_sec < block.start_sec or event.time_sec >= block.end_sec:
            continue
        if event.plugin == "sysmon" and len(event.command) == 2 and "failure" in event.command[0]:
            task_events["sysmon"].append(event)
        if event.plugin == "communications" and event.command and event.command[0] == "radioprompt":
            task_events["communications"].append(event)
        if event.plugin == "resman" and len(event.command) == 2 and "pump" in event.command[0] and event.command[1] == "failure":
            task_events["resman"].append(event)
    return task_events


def _check_block_schedule(path: Path, events: list[Event], blocks: list[Block], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for block in blocks:
        counts = LEVEL_COUNTS[block.level]
        task_events = _block_task_events(events, block)

        # Count checks
        for task, expected_count in counts.items():
            actual_count = len(task_events[task])
            if actual_count != expected_count:
                evidence.append(
                    Evidence(
                        f"{block.name} {block.level} {task} count {actual_count} != {expected_count}",
                        _rel_link(repo_root, path, block.start_line),
                    )
                )

        # Offset checks
        offsets: dict[str, list[int]] = {}
        for task, task_list in task_events.items():
            offsets[task] = [e.time_sec - block.start_sec for e in task_list]
            for offset, event in zip(offsets[task], task_list):
                if offset < 15 or offset > 289:
                    evidence.append(
                        Evidence(
                            f"{block.name} {task} offset {offset} out of bounds",
                            _rel_link(repo_root, path, event.line_no),
                        )
                    )

        # Overlap rule
        used: dict[int, list[str]] = {}
        for task, offset_list in offsets.items():
            for offset in offset_list:
                used.setdefault(offset, []).append(task)
        overlaps = {k: v for k, v in used.items() if len(set(v)) > 1}
        for offset, tasks in overlaps.items():
            evidence.append(
                Evidence(
                    f"{block.name} overlap at {offset}s: {sorted(set(tasks))}",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

        # Communications spacing and target/distractor assignment
        comm_events = task_events["communications"]
        comm_offsets = [e.time_sec - block.start_sec for e in comm_events]
        for prev, curr in zip(comm_offsets, comm_offsets[1:]):
            if curr - prev < 8:
                evidence.append(
                    Evidence(
                        f"{block.name} communications spacing {prev}->{curr}",
                        _rel_link(repo_root, path, comm_events[0].line_no) if comm_events else _rel_link(repo_root, path, block.start_line),
                    )
                )
        for idx, event in enumerate(comm_events, start=1):
            expected = "other" if idx % 5 == 0 else "own"
            actual = event.command[1] if len(event.command) > 1 else ""
            if actual != expected:
                evidence.append(
                    Evidence(
                        f"{block.name} comm #{idx} expected {expected} got {actual}",
                        _rel_link(repo_root, path, event.line_no),
                    )
                )

        # Deterministic schedule check
        base_offsets = _ordered_base_offsets(counts)
        try:
            assigned = _assign_offsets(base_offsets)
        except ValueError as exc:
            evidence.append(
                Evidence(
                    f"{block.name} deterministic assignment failed: {exc}",
                    _rel_link(repo_root, path, block.start_line),
                )
            )
            continue

        for task in ("sysmon", "communications", "resman"):
            expected_offsets = assigned[task]
            actual_offsets = offsets[task]
            if expected_offsets != actual_offsets:
                evidence.append(
                    Evidence(
                        f"{block.name} {task} offsets mismatch expected {expected_offsets[:5]}... got {actual_offsets[:5]}...",
                        _rel_link(repo_root, path, block.start_line),
                    )
                )

    return evidence


def _check_session_structure(path: Path, blocks: list[Block], events: list[Event], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []

    training = [b for b in blocks if b.kind == "training"]
    retained = [b for b in blocks if b.kind == "retained"]

    # Training order and durations
    expected_training = ["T1", "T2", "T3"]
    if [b.name for b in training] != expected_training:
        evidence.append(Evidence("Training order mismatch", _rel_link(repo_root, path, training[0].start_line if training else 1)))
    for block in training:
        duration = block.end_sec - block.start_sec
        if duration != BLOCK_DURATION_SEC:
            evidence.append(
                Evidence(
                    f"{block.name} duration {duration}s != 300s",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

    # Breaks between training blocks
    for prev, nxt in zip(training, training[1:]):
        gap = nxt.start_sec - prev.end_sec
        if gap != 60:
            evidence.append(
                Evidence(
                    f"Training break {prev.name}->{nxt.name} {gap}s != 60s",
                    _rel_link(repo_root, path, nxt.start_line),
                )
            )

    # Retained blocks duration and breaks
    for block in retained:
        duration = block.end_sec - block.start_sec
        if duration != BLOCK_DURATION_SEC:
            evidence.append(
                Evidence(
                    f"{block.name} duration {duration}s != 300s",
                    _rel_link(repo_root, path, block.start_line),
                )
            )

    # TLX immediately after retained block end
    marker_events = _find_marker_events(events)
    tlx_start = {e.time_sec: e for e in marker_events if _marker_name(e.command[-1]) and _marker_name(e.command[-1]).startswith("TLX/") and _marker_name(e.command[-1]).endswith("/START")}
    tlx_end = {e.time_sec: e for e in marker_events if _marker_name(e.command[-1]) and _marker_name(e.command[-1]).startswith("TLX/") and _marker_name(e.command[-1]).endswith("/END")}
    for block in retained:
        if block.end_sec not in tlx_start:
            evidence.append(
                Evidence(
                    f"{block.name} missing TLX START at block end",
                    _rel_link(repo_root, path, block.end_line),
                )
            )
        if block.end_sec not in tlx_end:
            evidence.append(
                Evidence(
                    f"{block.name} missing TLX END at block end",
                    _rel_link(repo_root, path, block.end_line),
                )
            )

    # Breaks between retained blocks (1:00 after TLX)
    retained_sorted = sorted(retained, key=lambda b: b.start_sec)
    for prev, nxt in zip(retained_sorted, retained_sorted[1:]):
        gap = nxt.start_sec - prev.end_sec
        if gap != 60:
            evidence.append(
                Evidence(
                    f"Retained break {prev.name}->{nxt.name} {gap}s != 60s",
                    _rel_link(repo_root, path, nxt.start_line),
                )
            )

    return evidence


def _check_markers_against_spec(path: Path, events: list[Event], seq_id: str, repo_root: Path) -> tuple[list[Evidence], list[Evidence]]:
    failures: list[Evidence] = []
    evidence: list[Evidence] = []
    expected = _expected_markers(seq_id)
    marker_events = _find_marker_events(events)
    payloads = {e.command[-1]: e for e in marker_events}
    for marker_text in expected:
        if marker_text not in payloads:
            failures.append(Evidence(f"Missing marker {marker_text}", _rel_link(repo_root, path, 1)))
        else:
            event = payloads[marker_text]
            evidence.append(Evidence(f"Marker present {marker_text}", _rel_link(repo_root, path, event.line_no)))

    # Check retained level tokens
    for event in marker_events:
        payload = event.command[-1]
        name = _marker_name(payload)
        if not name or not name.startswith("RETAINED/"):
            continue
        parts = name.split("/")
        if len(parts) == 4:
            level = parts[2]
            if level not in ALLOWED_LEVELS:
                failures.append(
                    Evidence(
                        f"Invalid retained level {level}",
                        _rel_link(repo_root, path, event.line_no),
                    )
                )

    return failures, evidence


def _check_instruction_assets(path: Path, repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    content = _read_text(path)
    if "instructions;filename" in content:
        evidence.append(Evidence("Scenario references instructions", _rel_link(repo_root, path, 1)))
    if re.search(r"default/|fr_FR|_fr\\.txt", content, flags=re.IGNORECASE):
        evidence.append(Evidence("Scenario references disallowed assets", _rel_link(repo_root, path, 1)))
    if "nasatlx_en.txt" not in content:
        evidence.append(Evidence("Scenario missing NASA-TLX asset", _rel_link(repo_root, path, 1)))
    if not evidence:
        evidence.append(Evidence("No disallowed instruction assets found", _rel_link(repo_root, path, 1)))
    return evidence


def _check_training_identical(scenario_paths: list[Path], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    segments: dict[Path, str] = {}
    for path in scenario_paths:
        lines = _read_text(path).splitlines()
        start = end = None
        for idx, line in enumerate(lines):
            if "TRAINING/T1/START" in line:
                start = idx
            if "TRAINING/T3/END" in line:
                end = idx
        if start is None or end is None:
            evidence.append(Evidence("Training markers missing", _rel_link(repo_root, path, 1)))
            continue
        segments[path] = "\n".join(lines[start : end + 1])
    if len({v for v in segments.values()}) != 1:
        evidence.append(Evidence("Training segments differ across scenarios", _rel_link(repo_root, scenario_paths[0], 1)))
    else:
        evidence.append(Evidence("Training segments identical across SEQ1-SEQ3", _rel_link(repo_root, scenario_paths[0], 1)))
    return evidence


def _check_wrapper_mapping(wrapper_path: Path, filenames: list[str], repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    if not wrapper_path.exists():
        evidence.append(Evidence("Wrapper not found", _rel_link(repo_root, wrapper_path)))
        return evidence
    wrapper_text = _read_text(wrapper_path)
    for name in filenames:
        if name not in wrapper_text:
            evidence.append(Evidence(f"Wrapper missing scenario mapping {name}", _rel_link(repo_root, wrapper_path, 1)))
        else:
            line_no = next((idx for idx, line in enumerate(wrapper_text.splitlines(), start=1) if name in line), 1)
            evidence.append(Evidence(f"Wrapper mapping {name}", _rel_link(repo_root, wrapper_path, line_no)))
    return evidence


def _check_manifest_requirements(
    output_root: Path,
    participant: str,
    session: str,
    seq_id: str,
    dry_run: bool,
    expected_scenario: str,
    repo_root: Path,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    manifest_path = _find_manifest(output_root, participant, session)
    if manifest_path is None:
        evidence.append(Evidence("No manifest found", None))
        return evidence

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("unattended") is not True:
        evidence.append(Evidence("Manifest unattended != true", manifest_path.as_posix()))
    if manifest.get("dry_run") is not dry_run:
        evidence.append(Evidence("Manifest dry_run mismatch", manifest_path.as_posix()))
    if manifest.get("seq_id") != seq_id:
        evidence.append(Evidence("Manifest seq_id mismatch", manifest_path.as_posix()))
    if manifest.get("scenario_name") != expected_scenario:
        evidence.append(Evidence("Manifest scenario_name mismatch", manifest_path.as_posix()))
    if not manifest.get("repo_commit") or not manifest.get("submodule_commit"):
        evidence.append(Evidence("Manifest missing commit metadata", manifest_path.as_posix()))

    output_dir = Path(manifest.get("output_dir", ""))
    if output_dir and repo_root in output_dir.parents:
        evidence.append(Evidence("Output directory inside repo", manifest_path.as_posix()))

    if not evidence:
        evidence.append(Evidence(f"Manifest OK {manifest_path}", manifest_path.as_posix()))
    return evidence


def _check_dry_run_csv(
    output_root: Path,
    participant: str,
    session: str,
    seq_id: str,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    manifest_path = _find_manifest(output_root, participant, session)
    if manifest_path is None:
        evidence.append(Evidence("No manifest found for CSV checks", None))
        return evidence
    csv_path = _find_latest_csv(manifest_path)
    if csv_path is None:
        evidence.append(Evidence("No CSV found for CSV checks", None))
        return evidence

    markers = _extract_markers_from_csv(csv_path)
    expected = _expected_dry_run_markers(participant, session, seq_id)
    for marker_text in expected:
        if marker_text not in markers:
            evidence.append(Evidence(f"Missing CSV marker {marker_text}", csv_path.as_posix()))

    if not evidence:
        evidence.append(Evidence(f"CSV markers OK {csv_path}", csv_path.as_posix()))
    return evidence


def _check_spec_files_exist(repo_root: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for rel in REPO_SPEC_FILES:
        path = repo_root / rel
        if not path.exists():
            evidence.append(Evidence(f"Missing spec file {rel}", None))
        else:
            evidence.append(Evidence(f"Found spec file {rel}", _rel_link(repo_root, path, 1)))
    return evidence


def _summarize(results: list[RequirementResult]) -> int:
    print("VERIFICATION REPORT")
    print("-------------------")
    for result in results:
        print(f"[{result.status}] {result.req_id} {result.category} :: {result.requirement}")
        for ev in result.evidence:
            if ev.link:
                print(f"  - {ev.label} :: {ev.link}")
            else:
                print(f"  - {ev.label}")
        if result.rationale:
            print(f"  - RATIONALE: {result.rationale}")
    any_fail = any(r.status == "FAIL" for r in results)
    return 1 if any_fail else 0


def check_unattended_manifest(
    output_root: Path,
    participant: str,
    session: str,
    seq_id: str,
    dry_run: bool,
    expected_scenario: str,
) -> list[str]:
    errors: list[str] = []
    manifest_path = _find_manifest(output_root, participant, session)
    if manifest_path is None:
        return ["No manifest found for the specified output root/participant/session."]

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("unattended") is not True:
        errors.append(f"Manifest missing unattended=true: {manifest_path}")
    if manifest.get("dry_run") is not dry_run:
        errors.append(f"Manifest dry_run mismatch (expected {dry_run}): {manifest_path}")
    if manifest.get("seq_id") != seq_id:
        errors.append(f"Manifest seq_id mismatch (expected {seq_id}): {manifest_path}")
    scenario_name = manifest.get("scenario_name")
    if scenario_name != expected_scenario:
        errors.append(f"Manifest scenario_name mismatch (expected {expected_scenario}): {manifest_path}")
    if not manifest.get("repo_commit") or not manifest.get("submodule_commit"):
        errors.append(f"Manifest missing repo/submodule commit metadata: {manifest_path}")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(manifest.get("output_dir", ""))
    if output_dir and repo_root in output_dir.parents:
        errors.append(f"Output dir is inside repo (violates data policy): {output_dir}")

    return errors


def check_dry_run_csv(
    output_root: Path,
    participant: str,
    session: str,
    seq_id: str,
) -> list[str]:
    errors: list[str] = []
    manifest_path = _find_manifest(output_root, participant, session)
    if manifest_path is None:
        return ["No manifest found for dry-run CSV checks."]
    csv_path = _find_latest_csv(manifest_path)
    if csv_path is None:
        return ["No session CSV found for dry-run CSV checks."]

    markers = _extract_markers_from_csv(csv_path)
    expected = _expected_dry_run_markers(participant, session, seq_id)
    for marker_text in expected:
        if marker_text not in markers:
            errors.append(f"Missing marker in CSV: {marker_text}")

    # Order check: expected markers appear in sequence
    last_index = -1
    for marker_text in expected:
        try:
            idx = markers.index(marker_text)
        except ValueError:
            errors.append(f"Marker missing for order check: {marker_text}")
            continue
        if idx < last_index:
            errors.append(f"Marker order violation: {marker_text}")
        last_index = idx

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pilot scenario artifacts and optional unattended outputs.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--seq-id", choices=("SEQ1", "SEQ2", "SEQ3"), default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-scenario", default="pilot_dry_run_v0.txt")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scenario_dir = repo_root / "src" / "python" / "vendor" / "openmatb" / "includes" / "scenarios"
    scenario_paths = [
        scenario_dir / "pilot_seq1.txt",
        scenario_dir / "pilot_seq2.txt",
        scenario_dir / "pilot_seq3.txt",
    ]
    dry_run_path = scenario_dir / args.dry_run_scenario
    wrapper_path = repo_root / "src" / "python" / "run_openmatb.py"

    results: list[RequirementResult] = []

    # Spec files presence
    spec_evidence = _check_spec_files_exist(repo_root)
    if any("Missing" in e.label for e in spec_evidence):
        _add_fail(results, "A1", "Artifact presence & naming", "Spec/contract files exist", spec_evidence)
    else:
        _add_pass(results, "A1", "Artifact presence & naming", "Spec/contract files exist", spec_evidence)

    # Scenario files presence
    scenario_evidence: list[Evidence] = []
    missing = False
    for path in scenario_paths + [dry_run_path]:
        if not path.exists():
            scenario_evidence.append(Evidence(f"Missing scenario {path}", None))
            missing = True
        else:
            scenario_evidence.append(Evidence(f"Found scenario {path.name}", _rel_link(repo_root, path, 1)))
    if missing:
        _add_fail(results, "A2", "Artifact presence & naming", "Scenario files exist", scenario_evidence)
    else:
        _add_pass(results, "A2", "Artifact presence & naming", "Scenario files exist", scenario_evidence)

    # Wrapper mapping
    wrapper_evidence = _check_wrapper_mapping(wrapper_path, [p.name for p in scenario_paths + [dry_run_path]], repo_root)
    if any("missing" in e.label.lower() for e in wrapper_evidence):
        _add_fail(results, "A3", "Artifact presence & naming", "Wrapper references all scenarios", wrapper_evidence)
    else:
        _add_pass(results, "A3", "Artifact presence & naming", "Wrapper references all scenarios", wrapper_evidence)

    # Training identical
    training_evidence = _check_training_identical(scenario_paths, repo_root)
    if any("differ" in e.label for e in training_evidence):
        _add_fail(results, "A4", "Artifact presence & naming", "Training segments identical across SEQ1-SEQ3", training_evidence)
    else:
        _add_pass(results, "A4", "Artifact presence & naming", "Training segments identical across SEQ1-SEQ3", training_evidence)

    # Scenario grammar and parser compliance
    grammar_failures: list[Evidence] = []
    payload_failures: list[Evidence] = []
    parser_failures: list[Evidence] = []
    parser_evidence: list[Evidence] = []
    for path in scenario_paths + [dry_run_path]:
        events = _parse_events(path)
        grammar_failures.extend(_check_scenario_grammar(path, events, repo_root))
        payload_failures.extend(_check_marker_payloads(path, events, repo_root))
        failures, evidence = _check_plugin_start_stop(path, events, repo_root)
        parser_failures.extend(failures)
        parser_evidence.extend(evidence)

    if grammar_failures:
        _add_fail(results, "B1", "Scenario grammar & parser compliance", "OpenMATB line format and time format", grammar_failures)
    else:
        _add_pass(results, "B1", "Scenario grammar & parser compliance", "OpenMATB line format and time format", [Evidence("All lines valid", _rel_link(repo_root, scenario_paths[0], 1))])

    if parser_failures:
        _add_fail(results, "B2", "Scenario grammar & parser compliance", "Plugins have start/stop as required", parser_failures)
    else:
        _add_pass(results, "B2", "Scenario grammar & parser compliance", "Plugins have start/stop as required", parser_evidence or [Evidence("All required plugin start/stop found", _rel_link(repo_root, scenario_paths[0], 1))])

    if payload_failures:
        _add_fail(results, "D1", "Marker presence, naming, ordering", "Marker payload format and token presence", payload_failures)
    else:
        _add_pass(results, "D1", "Marker presence, naming, ordering", "Marker payload format and token presence", [Evidence("All marker payloads valid", _rel_link(repo_root, scenario_paths[0], 1))])

    # Session structure semantics
    structure_failures: list[Evidence] = []
    for path in scenario_paths:
        events = _parse_events(path)
        blocks = _collect_blocks(events)
        structure_failures.extend(_check_session_structure(path, blocks, events, repo_root))
    if structure_failures:
        _add_fail(results, "C1", "Session structure semantics", "Training/retained duration and break timing", structure_failures)
    else:
        _add_pass(results, "C1", "Session structure semantics", "Training/retained duration and break timing", [Evidence("All blocks and breaks align with spec", _rel_link(repo_root, scenario_paths[0], 1))])

    # Markers list and levels
    marker_failures: list[Evidence] = []
    marker_evidence: list[Evidence] = []
    for seq_id, path in zip(("SEQ1", "SEQ2", "SEQ3"), scenario_paths):
        events = _parse_events(path)
        failures, evidence = _check_markers_against_spec(path, events, seq_id, repo_root)
        marker_failures.extend(failures)
        marker_evidence.extend(evidence)
    if marker_failures:
        _add_fail(results, "D2", "Marker presence, naming, ordering", "Marker list matches spec", marker_failures)
    else:
        _add_pass(results, "D2", "Marker presence, naming, ordering", "Marker list matches spec", marker_evidence or [Evidence("All required markers found", _rel_link(repo_root, scenario_paths[0], 1))])

    # Determinism & scheduling rules
    scheduling_failures: list[Evidence] = []
    for path in scenario_paths:
        events = _parse_events(path)
        blocks = _collect_blocks(events)
        scheduling_failures.extend(_check_block_schedule(path, events, blocks, repo_root))
    if scheduling_failures:
        _add_fail(results, "E1", "Determinism & scheduling rules", "Per-block deterministic schedule + overlap rules", scheduling_failures)
    else:
        _add_pass(results, "E1", "Determinism & scheduling rules", "Per-block deterministic schedule + overlap rules", [Evidence("All block schedules match deterministic template", _rel_link(repo_root, scenario_paths[0], 1))])

    # Instruction assets policy
    instruction_failures: list[Evidence] = []
    for path in scenario_paths:
        instruction_failures.extend(_check_instruction_assets(path, repo_root))
    if any("Scenario references" in e.label or "missing" in e.label.lower() for e in instruction_failures):
        _add_fail(results, "F1", "Instruction asset policy", "No disallowed instruction assets", instruction_failures)
    else:
        _add_pass(results, "F1", "Instruction asset policy", "No disallowed instruction assets", instruction_failures)

    # Data-management boundaries & manifest provenance
    if args.output_root and args.participant and args.session and args.seq_id:
        expected_scenario = args.dry_run_scenario.replace(".txt", "") if args.dry_run else f"pilot_seq{args.seq_id[-1]}"
        manifest_evidence = _check_manifest_requirements(
            args.output_root,
            args.participant,
            args.session,
            args.seq_id,
            args.dry_run,
            expected_scenario,
            repo_root,
        )
        if any("mismatch" in e.label.lower() or "inside" in e.label.lower() or "missing" in e.label.lower() for e in manifest_evidence):
            _add_fail(results, "G1", "Data-management boundaries & manifest provenance", "Manifest metadata and output location", manifest_evidence)
        else:
            _add_pass(results, "G1", "Data-management boundaries & manifest provenance", "Manifest metadata and output location", manifest_evidence)

        if args.dry_run:
            csv_evidence = _check_dry_run_csv(args.output_root, args.participant, args.session, args.seq_id)
            if any("Missing" in e.label for e in csv_evidence):
                _add_fail(results, "D3", "Marker presence, naming, ordering", "CSV marker presence (dry-run)", csv_evidence)
            else:
                _add_pass(results, "D3", "Marker presence, naming, ordering", "CSV marker presence (dry-run)", csv_evidence)
    else:
        _add_nmv(
            results,
            "G1",
            "Data-management boundaries & manifest provenance",
            "Manifest metadata and output location",
            "Requires runtime outputs and manifest path; run with --output-root/--participant/--session/--seq-id.",
        )
        _add_nmv(
            results,
            "D3",
            "Marker presence, naming, ordering",
            "CSV marker presence (dry-run)",
            "Requires runtime CSV output; run with --output-root/--participant/--session/--seq-id and --dry-run.",
        )

    # Explicit non-machine-verifiable requirements
    _add_nmv(
        results,
        "N1",
        "Session structure semantics",
        "Participant ID entry popup blocks start until valid ID submitted",
        "Requires UI interaction and human input; not verifiable statically.",
    )
    _add_nmv(
        results,
        "N2",
        "Marker transport requirements",
        "LSL marker presence when LSL is chosen timebase",
        "Requires runtime LSL stream inspection and external timebase selection.",
    )
    _add_nmv(
        results,
        "N3",
        "Marker set",
        "ABORT marker emitted only on early termination",
        "Runtime condition-dependent; cannot be verified statically without forced abort scenario.",
    )
    _add_nmv(
        results,
        "N4",
        "Session structure semantics",
        "NASA-TLX self-paced completion with all sliders interacted",
        "Requires participant interaction; not statically verifiable.",
    )
    _add_nmv(
        results,
        "N5",
        "Provisional alignment tolerance",
        "EEG/OpenMATB alignment tolerance ≤ 20 ms",
        "Requires EEG acquisition stack and timing measurements.",
    )
    _add_nmv(
        results,
        "N6",
        "Pause policy",
        "No pause during retained blocks; abort on interruption",
        "Operational procedure requirement; not detectable statically.",
    )

    return _summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
