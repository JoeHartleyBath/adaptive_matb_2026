"""generate_full_study_scenarios.py

Generates a 9-minute continuous MATB calibration scenario file for the full
study (post-Pilot 1), based on a participant's converged staircase difficulty
value (d_final).

The scenario contains nine 1-minute blocks: 3×LOW, 3×MODERATE, and 3×HIGH.
Difficulty transitions are issued as in-scenario parameter-update commands at
each 1-minute boundary — all tasks run continuously, with no pauses or
restarts between blocks.

Usage
-----
    # From a staircase-calibration session CSV:
    python scripts/generate_full_study_scenarios.py \\
        --participant P001 \\
        --condition 1 \\
        --calibration-csv "D:/data/openmatb/P001/session1.csv"

    # Override d_final directly (for testing):
    python scripts/generate_full_study_scenarios.py \\
        --participant P001 --condition 1 --d-final 0.55

    # Dry run (print to stdout, do not write file):
    python scripts/generate_full_study_scenarios.py \\
        --participant P001 --condition 1 --d-final 0.55 --dry-run

Level anchors
-------------
    d_MODERATE = d_final               (calibrated 70%-performance threshold)
    d_LOW      = max(0.0, d_final - delta)
    d_HIGH     = d_final + delta   (no ceiling — may exceed 1.0 at ceiling participants)
    default delta = 0.20  (= 4× staircase fine step of 0.05)

Counterbalancing
----------------
Six balanced-thirds templates are defined as module-level constants (T1–T6).
Each template contains exactly 3×LOW, 3×MODERATE, and 3×HIGH, with each level
appearing exactly once in blocks 1–3, 4–6, and 7–9.

    condition 1  →  TEMPLATES[(participant_rank + 0) % 6]
    condition 2  →  TEMPLATES[(participant_rank + 3) % 6]  (complementary)

participant_rank is the 0-based position in config/participant_assignments.yaml.

See also
--------
    docs/decisions/design_choices/study_design/dc_full_study_calibration_structure.md
    src/python/adaptation/difficulty_state.py  (canonical parameter mapping)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup: allow importing from src/ without installation
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptation.difficulty_state import make_task_params  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_DURATION_SEC: int = 60         # duration of each 1-minute block
N_BLOCKS: int = 9                    # total blocks per 9-minute scenario
SCENARIO_DURATION_SEC: int = BLOCK_DURATION_SEC * N_BLOCKS  # 540 s

# Seconds of silent pre-task time before block_01 START marker.
# The LSL plugin starts at t=0, but LabRecorder needs ~5s to discover
# and subscribe to the OpenMATB marker stream.  By delaying all task
# content and markers by LSL_SETTLE_SEC, block_01 START is captured
# reliably, giving it the same labelled duration as every other block.
LSL_SETTLE_SEC: int = 5

EVENT_EDGE_BUFFER_SEC: int = 3       # guard zone at each edge within a block

DEFAULT_DELTA: float = 0.80          # low/high offset from d_final
DEFAULT_OUTPUT_DIR: Path = _REPO_ROOT / "experiment" / "scenarios"
CONFIG_PATH: Path = _REPO_ROOT / "config" / "participant_assignments.yaml"

# Ported from generate_pilot_scenarios.py / vendor defaults
AVERAGE_AUDITORY_PROMPT_DURATION_SEC: int = 18
COMMUNICATIONS_REFRACTORY_SEC: int = 1
RESMAN_PUMP_FAILURE_DURATION_SEC: int = 10
RESMAN_PUMP_REFRACTORY_SEC: int = 1
SYSMON_ALERT_TIMEOUT_SEC: int = 10
SYSMON_REFRACTORY_SEC: int = 1

# Pump flow rates (from vendor resman.py defaults)
RESMAN_PUMPS: dict[str, int] = {
    "1": 800, "2": 600, "3": 800, "4": 600,
    "5": 600, "6": 600, "7": 400, "8": 400,
}

SYSMON_LIGHTS: list[str] = ["1", "2"]
SYSMON_SCALES: list[str] = ["1", "2", "3", "4"]

# ---------------------------------------------------------------------------
# Counterbalancing templates
# ---------------------------------------------------------------------------
# Six balanced-thirds templates.  Each contains exactly 3×L, 3×M, 3×H.
# Each level appears exactly once in positions 0–2, 3–5, and 6–8.
# No run of 3+ identical consecutive levels.
#
# Conditions are paired so that for any participant, conditions 1 and 2
# use complementary orderings:
#     (T1, T4), (T2, T5), (T3, T6)

TEMPLATES: list[list[str]] = [
    # T1
    ["L", "M", "H",  "M", "H", "L",  "H", "L", "M"],
    # T2
    ["L", "H", "M",  "H", "M", "L",  "M", "L", "H"],
    # T3
    ["M", "L", "H",  "L", "H", "M",  "H", "M", "L"],
    # T4
    ["M", "H", "L",  "H", "L", "M",  "L", "M", "H"],
    # T5
    ["H", "L", "M",  "L", "M", "H",  "M", "H", "L"],
    # T6
    ["H", "M", "L",  "M", "L", "H",  "L", "H", "M"],
]

_LEVEL_FULL: dict[str, str] = {"L": "LOW", "M": "MODERATE", "H": "HIGH"}


# ---------------------------------------------------------------------------
# d_final extraction
# ---------------------------------------------------------------------------

def extract_d_final(csv_path: Path) -> float:
    """Extract the converged staircase difficulty from an OpenMATB session CSV.

    Priority order:
    1. Last ``adaptation_converged`` event  (clean convergence).
    2. Last ``adaptation_step`` d value with a warning  (timed out mid-staircase).
    3. d_init from ``adaptation_init`` with a warning  (scenario timed out before
       any step fired — participant was already on-target at the starting difficulty).

    Raises ``ValueError`` if no adaptation rows are present at all, or
    ``FileNotFoundError`` if the path does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Session CSV not found: {csv_path}")

    converged_d: Optional[float] = None
    last_step_d: Optional[float] = None
    d_init: Optional[float] = None

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("type") != "adaptation":
                continue
            try:
                data: dict = json.loads(row.get("value", ""))
            except (json.JSONDecodeError, TypeError):
                continue

            event = data.get("event", "")
            if event == "adaptation_converged":
                converged_d = float(data["d"])
            elif event == "adaptation_step":
                state = data.get("state", {})
                if "d" in state:
                    last_step_d = float(state["d"])
            elif event == "adaptation_init":
                cfg = data.get("config", {})
                if "d_init" in cfg:
                    d_init = float(cfg["d_init"])

    if converged_d is not None:
        return converged_d

    if last_step_d is not None:
        warnings.warn(
            f"No 'adaptation_converged' event found in {csv_path.name}. "
            f"Using last adaptation_step d={last_step_d:.4f} as d_final. "
            "The staircase may not have fully converged.",
            UserWarning,
            stacklevel=2,
        )
        return last_step_d

    if d_init is not None:
        warnings.warn(
            f"No adaptation steps found in {csv_path.name}. "
            f"The staircase scenario timed out without firing a single step — "
            f"performance was on-target at d_init from the start. "
            f"Using d_init={d_init:.4f} as d_final.",
            UserWarning,
            stacklevel=2,
        )
        return d_init

    raise ValueError(
        f"No adaptation events found in {csv_path}. "
        "Confirm this is an adaptation-session CSV."
    )


# ---------------------------------------------------------------------------
# Level anchor computation
# ---------------------------------------------------------------------------

def compute_level_difficulties(
    d_final: float,
    delta: float = DEFAULT_DELTA,
) -> dict[str, float]:
    """Derive per-level difficulty anchors from d_final.

    Returns a dict with keys "LOW", "MODERATE", "HIGH".

    No clamping is applied to either anchor:
      d_LOW  = d_final - delta  (may be < 0 for easy participants)
      d_HIGH = d_final + delta  (may be > 1 for ceiling participants)

    Both cases extrapolate linearly via make_task_params(), which applies
    physical floors/ceilings to keep all task parameters valid.  Warnings are
    emitted so the researcher is aware of extrapolation on either end.
    """
    d_low  = d_final - delta
    d_mid  = float(d_final)
    d_high = d_final + delta

    if d_low < 0.0:
        warnings.warn(
            f"d_final={d_final:.4f}: LOW anchor d_low={d_low:.4f} is below 0. "
            "Task parameters extrapolate below the easy end: resman drain = 0, "
            "slower tracking update, fewer sysmon/comms events. "
            "This is valid for easy participants.",
            UserWarning,
            stacklevel=2,
        )
    if d_high > 1.0:
        warnings.warn(
            f"d_final={d_final:.4f}: HIGH anchor d_high={d_high:.4f} exceeds 1.0. "
            "Task parameters extrapolate above the hard end: higher resman drain, "
            "denser sysmon/comms events, faster tracking. "
            "This is intentional for ceiling participants.",
            UserWarning,
            stacklevel=2,
        )

    return {"LOW": d_low, "MODERATE": d_mid, "HIGH": d_high}


# ---------------------------------------------------------------------------
# Template assignment
# ---------------------------------------------------------------------------

def _participant_rank(participant_id: str) -> int:
    """Return participant's 0-based position in participant_assignments.yaml.

    Falls back to a hash-based value if the file cannot be read or the
    participant is not listed, so the script works standalone in testing.
    """
    try:
        import yaml  # type: ignore[import]
        with CONFIG_PATH.open(encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        ids = list(config.get("participants", {}).keys())
        if participant_id in ids:
            return ids.index(participant_id)
        warnings.warn(
            f"Participant '{participant_id}' not found in {CONFIG_PATH.name}; "
            "using hash-based template rank.",
            UserWarning,
            stacklevel=3,
        )
    except Exception:  # noqa: BLE001
        warnings.warn(
            f"Could not load {CONFIG_PATH}; using hash-based template rank.",
            UserWarning,
            stacklevel=3,
        )
    return abs(hash(participant_id)) % 6


def assign_template(participant_id: str, condition: int) -> list[str]:
    """Return the 9-block label sequence (full names) for a participant/condition.

    *condition* must be 1 or 2.
    """
    if condition not in (1, 2):
        raise ValueError(f"condition must be 1 or 2, got {condition!r}")
    rank = _participant_rank(participant_id)
    offset = 0 if condition == 1 else 3
    short = TEMPLATES[(rank + offset) % 6]
    return [_LEVEL_FULL[s] for s in short]


# ---------------------------------------------------------------------------
# Internal line representation
# ---------------------------------------------------------------------------

@dataclass
class _Line:
    """One scenario file line with a numeric (absolute) timestamp."""
    time_sec: float
    plugin: str
    command: str   # everything after the plugin field, e.g. "taskupdatetime;42"

    @property
    def text(self) -> str:
        h = int(self.time_sec // 3600)
        m = int((self.time_sec % 3600) // 60)
        s = int(self.time_sec % 60)
        return f"{h}:{m:02d}:{s:02d};{self.plugin};{self.command}"


# ---------------------------------------------------------------------------
# Event distribution utilities  (adapted from generate_pilot_scenarios.py)
# ---------------------------------------------------------------------------

def _split_durations(total: float, n: int, rng: random.Random) -> list[float]:
    """Split *total* seconds into *n* non-negative contiguous intervals."""
    if n <= 1:
        return [total]
    cuts = sorted(rng.uniform(0.0, total) for _ in range(n - 1))
    pts = [0.0] + cuts + [total]
    return [pts[i + 1] - pts[i] for i in range(n)]


def _distribute_events(
    lines: list[_Line],
    plugin: str,
    commands: list[str],
    single_duration_sec: float,
    block_start_sec: float,
    rng: random.Random,
) -> list[_Line]:
    """Place events within the buffered window of a block.

    Events are scheduled between (block_start + EDGE_BUFFER) and
    (block_start + BLOCK_DURATION - EDGE_BUFFER - single_duration).
    Excess events that do not fit are silently trimmed.
    """
    effective = BLOCK_DURATION_SEC - 2 * EVENT_EDGE_BUFFER_SEC
    if effective <= 0 or not commands:
        return lines

    if single_duration_sec > 0:
        max_fit = int(math.floor(effective / single_duration_sec))
        commands = commands[:max_fit]

    if not commands:
        return lines

    rest = max(0.0, effective - len(commands) * single_duration_sec)
    gaps = _split_durations(rest, len(commands) + 1, rng)

    cursor = block_start_sec + EVENT_EDGE_BUFFER_SEC
    for gap, cmd in zip(gaps[:-1], commands):
        cursor += gap
        lines.append(_Line(cursor, plugin, cmd))
        cursor += single_duration_sec

    return lines


def _distribute_pump_failures(
    lines: list[_Line],
    pump_ids: list[str],
    block_start_sec: float,
    rng: random.Random,
) -> list[_Line]:
    """Schedule pump failures, each followed by a recovery 10 s later."""
    slot = RESMAN_PUMP_FAILURE_DURATION_SEC + RESMAN_PUMP_REFRACTORY_SEC  # 11 s
    effective = BLOCK_DURATION_SEC - 2 * EVENT_EDGE_BUFFER_SEC

    if effective <= 0 or not pump_ids:
        return lines

    max_fit = int(math.floor(effective / slot)) if slot > 0 else 0
    pump_ids = pump_ids[:max_fit]

    if not pump_ids:
        return lines

    rest = max(0.0, effective - len(pump_ids) * slot)
    gaps = _split_durations(rest, len(pump_ids) + 1, rng)

    cursor = block_start_sec + EVENT_EDGE_BUFFER_SEC
    for gap, pid in zip(gaps[:-1], pump_ids):
        cursor += gap
        lines.append(_Line(cursor, "resman", f"pump-{pid}-state;failure"))
        lines.append(_Line(
            cursor + RESMAN_PUMP_FAILURE_DURATION_SEC,
            "resman", f"pump-{pid}-state;off",
        ))
        cursor += slot

    return lines


def _sample(items: list, k: int, rng: random.Random) -> list:
    """Draw k items from *items* with repetition, shuffled."""
    pool = list(items)
    result: list = []
    while len(result) < k:
        rng.shuffle(pool)
        result.extend(pool[:k - len(result)])
    rng.shuffle(result)
    return result


def _sample_weighted(
    items: list[str],
    weights: list[float],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Largest-remainder weighted allocation, then shuffle."""
    if k <= 0:
        return []
    total_w = sum(weights)
    if total_w <= 0:
        return _sample(items, k, rng)

    expected = [(item, (w / total_w) * k) for item, w in zip(items, weights)]
    counts = {item: int(math.floor(e)) for item, e in expected}
    remaining = k - sum(counts.values())

    remainders = [(item, e - math.floor(e)) for item, e in expected]
    rng.shuffle(remainders)
    remainders.sort(key=lambda x: x[1], reverse=True)
    for item, _ in remainders[:remaining]:
        counts[item] += 1

    out: list[str] = []
    for item in items:
        out.extend([item] * counts[item])
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Per-block line generation
# ---------------------------------------------------------------------------

def generate_block_lines(
    level_label: str,
    d: float,
    block_index: int,
    block_start_sec: int,
    participant_id: str,
    condition: int,
    rng: random.Random,
    *,
    is_first_block: bool,
    condition_prefix: str = "calibration_condition",
) -> list[_Line]:
    """Return all scenario lines for one 1-minute block.

    Parameters
    ----------
    level_label      : "LOW", "MODERATE", or "HIGH"
    d                : difficulty scalar for this level
    block_index      : 0-based index (0–8)
    block_start_sec  : absolute start time within the scenario (s)
    is_first_block   : emit plugin start commands when True (first block only)
    condition_prefix : first segment of the LSL marker path
                       (default: "calibration_condition", override for other
                       study conditions e.g. "adaptive_automation")
    """
    params = make_task_params(d)      # call directly so d > 1.0 is not clamped
    lines: list[_Line] = []
    t0 = float(block_start_sec)

    pid_tok = "${OPENMATB_PARTICIPANT}"
    sid_tok = "${OPENMATB_SESSION}"
    payload = f"pid={pid_tok}|sid={sid_tok}"

    block_num_str = f"{block_index + 1:02d}"
    marker_base = f"{condition_prefix}/{condition}/block_{block_num_str}/{level_label}"

    # Block-start LSL marker
    lines.append(_Line(t0, "labstreaminglayer",
                       f"marker;STUDY/V0/{marker_base}/START|{payload}"))

    # Parameter updates (issued at every block boundary, including block 0)
    track_ms    = int(round(params.track_update_ms))
    track_force = int(round(params.track_joystick_force))
    leak        = params.resman_loss_a_per_min  # int; same for A and B

    lines.append(_Line(t0, "track",  f"taskupdatetime;{track_ms}"))
    lines.append(_Line(t0, "track",  f"joystickforce;{track_force}"))
    lines.append(_Line(t0, "resman", f"tank-a-lossperminute;{leak}"))
    lines.append(_Line(t0, "resman", f"tank-b-lossperminute;{leak}"))

    if is_first_block:
        lines.append(_Line(t0, "track",          "start"))
        lines.append(_Line(t0, "sysmon",         "start"))
        lines.append(_Line(t0, "communications", "start"))
        lines.append(_Line(t0, "resman",         "start"))

    # ------------------------------------------------------------------ #
    # Demand events — counts derived from calibrated rates in TaskParams    #
    # (see difficulty_state.py for anchoring to pilot scenario data).       #
    #                                                                        #
    # SysMon alerts on independent channels can coexist, so they are        #
    # scheduled with SYSMON_REFRACTORY_SEC (1 s) spacing rather than the    #
    # full alert-timeout slot.  This lifts the previous hard cap of 4 events #
    # per group and lets event counts scale properly with d.                 #
    # Comms remains serial (one prompt at a time), so comms_slot is kept.   #
    # ------------------------------------------------------------------ #

    _eff = float(BLOCK_DURATION_SEC - 2 * EVENT_EDGE_BUFFER_SEC)  # 54 s

    # SysMon: lights + scales (independent channels, can overlap).
    n_lights = round(params.sysmon_light_rate_hz * _eff)
    n_scales = round(params.sysmon_scale_rate_hz * _eff)
    if n_lights > 0:
        light_cmds = [f"lights-{l}-failure;True"
                      for l in _sample(SYSMON_LIGHTS, n_lights, rng)]
        lines = _distribute_events(
            lines, "sysmon", light_cmds, SYSMON_REFRACTORY_SEC, t0, rng)

    if n_scales > 0:
        scale_cmds = [f"scales-{s}-failure;True"
                      for s in _sample(SYSMON_SCALES, n_scales, rng)]
        lines = _distribute_events(
            lines, "sysmon", scale_cmds, SYSMON_REFRACTORY_SEC, t0, rng)

    # Communications: one prompt at a time — comms_slot is the scheduling slot.
    comms_slot = AVERAGE_AUDITORY_PROMPT_DURATION_SEC + COMMUNICATIONS_REFRACTORY_SEC  # 19 s
    n_comms = round(params.comms_rate_hz * _eff)
    if n_comms > 0:
        n_own   = n_comms // 2
        n_other = n_comms - n_own
        prompt_types = ["own"] * n_own + ["other"] * n_other
        rng.shuffle(prompt_types)
        comms_cmds = [f"radioprompt;{p}" for p in prompt_types]
        lines = _distribute_events(
            lines, "communications", comms_cmds, comms_slot, t0, rng)

    # ResMan pump failures: derive count from calibrated rate.
    n_pump = round(params.resman_pump_rate_hz * _eff)
    if n_pump > 0:
        pump_ids = list(RESMAN_PUMPS.keys())
        pump_weights = [1.0 / float(RESMAN_PUMPS[p]) for p in pump_ids]
        chosen = _sample_weighted(pump_ids, pump_weights, n_pump, rng)
        lines = _distribute_pump_failures(lines, chosen, t0, rng)

    # Block-end LSL marker — fires 1 s before the next boundary
    lines.append(_Line(
        t0 + BLOCK_DURATION_SEC - 1,
        "labstreaminglayer",
        f"marker;STUDY/V0/{marker_base}/END|{payload}",
    ))

    return lines


# ---------------------------------------------------------------------------
# Scenario assembly and file output
# ---------------------------------------------------------------------------

def _fmt_t(total_sec: float) -> str:
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return f"{h}:{m:02d}:{s:02d}"


def write_scenario(
    output_path: Path,
    participant_id: str,
    condition: int,
    level_sequence: list[str],
    level_difficulties: dict[str, float],
    d_final: float,
    delta: float,
    *,
    dry_run: bool = False,
) -> None:
    """Assemble and write (or print) the full 9-minute scenario file."""

    pid_tok = "${OPENMATB_PARTICIPANT}"
    sid_tok = "${OPENMATB_SESSION}"
    payload = f"pid={pid_tok}|sid={sid_tok}"

    cond_marker = f"calibration_condition/{condition}"
    scene_start_m = f"STUDY/V0/{cond_marker}/START|{payload}"
    scene_end_m   = f"STUDY/V0/{cond_marker}/END|{payload}"

    # Generate all 9 blocks
    all_lines: list[_Line] = []
    for block_index, level_label in enumerate(level_sequence):
        d = level_difficulties[level_label]
        block_start = block_index * BLOCK_DURATION_SEC
        block_rng = random.Random(f"{participant_id}|{condition}|{block_index}")
        block_lines = generate_block_lines(
            level_label=level_label,
            d=d,
            block_index=block_index,
            block_start_sec=block_start,
            participant_id=participant_id,
            condition=condition,
            rng=block_rng,
            is_first_block=(block_index == 0),
        )
        all_lines.extend(block_lines)

    # Stable sort by time (insertion order preserved within same timestamp)
    all_lines.sort(key=lambda ln: ln.time_sec)

    # Shift all task content forward by LSL_SETTLE_SEC.  The preamble lines
    # (labstreaminglayer;start, scheduling;start, voiceidiom, voicegender)
    # remain at t=0 to configure plugins before task events fire.  This
    # gives LabRecorder time to subscribe to the OpenMATB marker stream so
    # that block_01 START is reliably captured, keeping all block durations equal.
    all_lines = [_Line(ln.time_sec + LSL_SETTLE_SEC, ln.plugin, ln.command)
                 for ln in all_lines]

    end_t = _fmt_t(SCENARIO_DURATION_SEC + LSL_SETTLE_SEC)
    level_ds_str = ", ".join(
        f"{lbl}={level_difficulties[lbl]:.4f}"
        for lbl in ("LOW", "MODERATE", "HIGH")
    )
    block_order_str = " ".join(
        seq[0] for seq in level_sequence  # L/M/H abbreviations
    )

    lines_out: list[str] = [
        f"# OpenMATB Scenario: {output_path.name}",
        f"# Full-study calibration -- condition {condition}, 9 min continuous",
        f"#",
        f"# Participant  : {participant_id}",
        f"# d_final      : {d_final:.4f}",
        f"# delta        : {delta:.4f}",
        f"# Level anchors: {level_ds_str}",
        f"# Block order  : {block_order_str}  (L=LOW M=MODERATE H=HIGH)",
        f"# Seed basis   : <participant>|<condition>|<block_index>",
        f"# Generated by : scripts/generate_full_study_scenarios.py",
        f"",
        f"0:00:00;labstreaminglayer;start",
        f"0:00:00;labstreaminglayer;marker;{scene_start_m}",
        f"0:00:00;scheduling;start",
        f"0:00:00;communications;voiceidiom;english",
        f"0:00:00;communications;voicegender;male",
    ]

    for ln in all_lines:
        lines_out.append(ln.text)

    lines_out += [
        f"",
        f"# --- Scenario end ---",
        f"{end_t};sysmon;stop",
        f"{end_t};track;stop",
        f"{end_t};communications;stop",
        f"{end_t};resman;stop",
        f"{end_t};scheduling;stop",
        f"{end_t};labstreaminglayer;marker;{scene_end_m}",
        f"",
        f"# --- NASA-TLX (blocking) ---",
        f"{end_t};labstreaminglayer;marker;STUDY/V0/TLX/{cond_marker}/START|{payload}",
        f"{end_t};genericscales;filename;nasatlx_en.txt",
        f"{end_t};genericscales;start",
        f"{end_t};labstreaminglayer;marker;STUDY/V0/TLX/{cond_marker}/END|{payload}",
        f"{end_t};labstreaminglayer;stop",
    ]

    content = "\n".join(lines_out) + "\n"

    if dry_run:
        print(content)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"Written: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 9-minute full-study calibration scenario "
            "from a participant's staircase d_final."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--participant", required=True,
        help="Participant ID (e.g. P001)",
    )
    parser.add_argument(
        "--condition", type=int, choices=[1, 2], required=True,
        help="Calibration condition number (1 or 2)",
    )

    d_source = parser.add_mutually_exclusive_group(required=True)
    d_source.add_argument(
        "--calibration-csv", type=Path, metavar="CSV",
        help="Path to the OpenMATB staircase-calibration session CSV",
    )
    d_source.add_argument(
        "--d-final", type=float, metavar="D",
        help="Override d_final directly, e.g. 0.55  (for testing)",
    )

    parser.add_argument(
        "--delta", type=float, default=DEFAULT_DELTA,
        help=f"Low/high offset from d_final (default: {DEFAULT_DELTA})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: scenarios/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print scenario to stdout instead of writing to disk",
    )

    args = parser.parse_args()

    # Resolve d_final
    if args.d_final is not None:
        d_final = args.d_final
        if not -0.8 <= d_final <= 1.8:
            parser.error(f"--d-final must be in [-0.8, 1.8], got {d_final}")
        print(f"d_final = {d_final:.4f}  (source: --d-final override)")
    else:
        d_final = extract_d_final(args.calibration_csv)
        print(f"d_final = {d_final:.4f}  (source: {args.calibration_csv.name})")

    level_difficulties = compute_level_difficulties(d_final, args.delta)
    print(
        f"Level anchors:  "
        f"LOW={level_difficulties['LOW']:.4f}  "
        f"MODERATE={level_difficulties['MODERATE']:.4f}  "
        f"HIGH={level_difficulties['HIGH']:.4f}"
    )

    level_sequence = assign_template(args.participant, args.condition)
    block_order_str = " ".join(s[0] for s in level_sequence)
    print(f"Block order (condition {args.condition}): {block_order_str}")

    # Print per-level parameter preview
    for label, d in level_difficulties.items():
        p = make_task_params(d)
        print(
            f"  {label:8s} d={d:.4f}  track_ms={int(round(p.track_update_ms))}"
            f"  joystick={int(round(p.track_joystick_force))}"
            f"  leak={p.resman_loss_a_per_min} ml/min"
        )

    filename = (
        f"full_calibration_{args.participant.lower()}"
        f"_c{args.condition}.txt"
    )
    output_path = args.output_dir / filename

    write_scenario(
        output_path=output_path,
        participant_id=args.participant,
        condition=args.condition,
        level_sequence=level_sequence,
        level_difficulties=level_difficulties,
        d_final=d_final,
        delta=args.delta,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\nQuick check (dry run for comparison):")
        print(
            f"  python scripts/generate_full_study_scenarios.py "
            f"--participant {args.participant} "
            f"--condition {args.condition} "
            f"--d-final {d_final:.4f} --dry-run"
        )


if __name__ == "__main__":
    main()
