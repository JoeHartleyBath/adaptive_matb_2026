"""Shared data structures and loaders for the MWL adaptation audit log.

Used by:
  - scripts/analysis/analyse_adaptation_session.py
  - scripts/analysis/plot_adaptation_session.py
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from adaptation.adaptation_logger import COLUMNS as AUDIT_COLUMNS

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuditRow:
    """One row from the adaptation audit CSV."""
    timestamp_lsl: float
    scenario_time_s: float
    mwl_raw: float
    mwl_smoothed: float
    signal_quality: float
    threshold: float
    action: str            # hold | assist_on | assist_off
    assistance_on: bool
    cooldown_remaining_s: float
    hold_counter_s: float
    reason: str


@dataclass
class BlockSegment:
    """One workload block parsed from OpenMATB session markers."""
    name: str              # e.g. "calibration_condition/1/block_01/LOW"
    level: str             # LOW | MODERATE | HIGH
    block_num: int         # 1-based block number
    start_sec: float
    end_sec: float


# Regex to extract the level label from a block marker segment name
# Pattern: .../block_NN/LEVEL
_LEVEL_RE = re.compile(r"/block_(\d+)/(\w+)$")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_audit_csv(path: Path) -> list[AuditRow]:
    """Load the adaptation audit CSV into typed rows."""
    rows: list[AuditRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            sys.exit(f"ERROR: Empty CSV: {path}")
        missing = set(AUDIT_COLUMNS) - set(reader.fieldnames)
        if missing:
            sys.exit(f"ERROR: Audit CSV missing columns: {missing}")

        for row in reader:
            rows.append(AuditRow(
                timestamp_lsl=float(row["timestamp_lsl"]),
                scenario_time_s=float(row["scenario_time_s"]),
                mwl_raw=float(row["mwl_raw"]),
                mwl_smoothed=float(row["mwl_smoothed"]),
                signal_quality=float(row["signal_quality"]),
                threshold=float(row["threshold"]),
                action=row["action"].strip(),
                assistance_on=row["assistance_on"].strip() == "True",
                cooldown_remaining_s=float(row["cooldown_remaining_s"]),
                hold_counter_s=float(row["hold_counter_s"]),
                reason=row["reason"],
            ))
    if not rows:
        sys.exit(f"ERROR: No data rows in audit CSV: {path}")
    return rows


def load_session_blocks(csv_path: Path) -> list[BlockSegment]:
    """Parse OpenMATB session CSV for workload block START/END markers.

    Expects markers of the form:
        STUDY/V0/{prefix}/{condition}/block_{NN}/{LEVEL}/START|...
        STUDY/V0/{prefix}/{condition}/block_{NN}/{LEVEL}/END|...
    """
    starts: dict[str, tuple[float, str, int]] = {}
    ends: dict[str, float] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip().lower()
            module = (row.get("module") or "").strip().lower()
            address = (row.get("address") or "").strip().lower()
            if row_type != "event" or module != "labstreaminglayer" or address != "marker":
                continue

            t_str = row.get("scenario_time") or ""
            try:
                t = float(t_str)
            except ValueError:
                continue

            raw_value = (row.get("value") or "").split("|", 1)[0].strip()
            if not raw_value:
                continue

            body = raw_value
            if body.startswith("STUDY/V0/"):
                body = body[len("STUDY/V0/"):]

            if body.endswith("/START"):
                base = body[:-len("/START")]
                m = _LEVEL_RE.search(base)
                if m:
                    block_num = int(m.group(1))
                    level = m.group(2).upper()
                    starts[base] = (t, level, block_num)
            elif body.endswith("/END"):
                base = body[:-len("/END")]
                ends[base] = t

    segments: list[BlockSegment] = []
    for base, (start_t, level, block_num) in starts.items():
        end_t = ends.get(base)
        if end_t is None or end_t <= start_t:
            continue
        segments.append(BlockSegment(
            name=base,
            level=level,
            block_num=block_num,
            start_sec=start_t,
            end_sec=end_t,
        ))

    segments.sort(key=lambda s: s.start_sec)
    return segments
