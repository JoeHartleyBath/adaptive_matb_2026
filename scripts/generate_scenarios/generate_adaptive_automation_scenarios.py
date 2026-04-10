"""generate_adaptive_automation_scenarios.py

Generates a pre-scheduled MATB scenario file for the adaptive-automation
condition, based on a participant's converged staircase difficulty value
(d_final).

The scenario contains N 1-minute blocks split 50/50 between a LOW/MODERATE
half and a HIGH half:

    8 blocks (default): 2×LOW + 2×MODERATE + 4×HIGH  →  8 minutes
    10 blocks         : 2×LOW + 3×MODERATE + 5×HIGH  → 10 minutes

Each block follows the same structure as the full-study calibration scenarios
(see generate_full_study_scenarios.py): difficulty-parameter updates at block
start, pre-scheduled sysmon / comms / resman events, and LSL markers.

Event generation is imported directly from generate_full_study_scenarios so
that the two scripts remain in sync.

Usage
-----
    # From a staircase-calibration session CSV:
    python scripts/generate_adaptive_automation_scenarios.py \\
        --participant P001 \\
        --condition 1 \\
        --calibration-csv "D:/data/openmatb/P001/session1.csv"

    # Override d_final directly (for testing / dry runs):
    python scripts/generate_adaptive_automation_scenarios.py \\
        --participant P001 --condition 1 --d-final 0.55

    # 10-block variant:
    python scripts/generate_adaptive_automation_scenarios.py \\
        --participant P001 --condition 1 --d-final 0.55 --n-blocks 10

    # Dry run (print to stdout, do not write file):
    python scripts/generate_adaptive_automation_scenarios.py \\
        --participant P001 --condition 1 --d-final 0.55 --dry-run

Level anchors
-------------
    d_MODERATE = d_final               (calibrated 70%-performance threshold)
    d_LOW      = max(0.0, d_final - delta)
    d_HIGH     = d_final + delta   (no ceiling — may exceed 1.0 at ceiling participants)
    default delta = 0.20

Counterbalancing
----------------
Six balanced templates are defined for each block count (8 and 10).
Each template preserves the 50/50 LOW-or-MODERATE / HIGH split and avoids
runs of more than two consecutive HIGH blocks.

    condition 1  →  TEMPLATES_8[rank % 6]       or TEMPLATES_10[rank % 6]
    condition 2  →  TEMPLATES_8[(rank + 3) % 6] or TEMPLATES_10[(rank + 3) % 6]

participant_rank is the 0-based position in config/participant_assignments.yaml.
(Conditions 1 and 2 use complementary pairs — same offset scheme as the
calibration generator.)

See also
--------
    docs/openmatb/ADAPTATION_DESIGN.md
    scripts/generate_full_study_scenarios.py   (calibration counterpart)
    src/adaptation/difficulty_state.py         (canonical parameter mapping)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# Re-use all event-generation machinery from the calibration generator so
# the two scripts stay in sync without code duplication.
from generate_full_study_scenarios import (  # noqa: E402
    BLOCK_DURATION_SEC,
    compute_level_difficulties,
    extract_d_final,
    generate_block_lines,
    _Line,
    _fmt_t,
)
from adaptation.difficulty_state import make_task_params  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_N_BLOCKS: tuple[int, ...] = (8, 10)
DEFAULT_N_BLOCKS: int = 8
DEFAULT_DELTA: float = 0.80
DEFAULT_OUTPUT_DIR: Path = _REPO_ROOT / "experiment" / "scenarios"
CONFIG_PATH: Path = _REPO_ROOT / "config" / "participant_assignments.yaml"

# ---------------------------------------------------------------------------
# Counterbalancing templates
# ---------------------------------------------------------------------------
# 8-block templates  (2×L + 2×M + 4×H)
# ----------------------------------------
# Six complementary pairs:  (T1,T4), (T2,T5), (T3,T6)
# No run of more than 2 consecutive identical levels.
#
#   pos:   0    1    2    3    4    5    6    7
# T1 =    L    H    M    H    H    L    H    M
# T2 =    M    H    L    H    H    M    H    L
# T3 =    H    L    H    H    M    H    L    M
# T4 =    H    M    H    H    L    H    M    L
# T5 =    L    H    H    M    H    L    H    M
# T6 =    M    H    H    L    H    M    H    L

TEMPLATES_8: list[list[str]] = [
    # T1
    ["LOW", "HIGH", "MODERATE", "HIGH", "HIGH", "LOW",      "HIGH", "MODERATE"],
    # T2  (complement of T1 — swap L↔M)
    ["MODERATE", "HIGH", "LOW",  "HIGH", "HIGH", "MODERATE", "HIGH", "LOW"],
    # T3
    ["HIGH", "LOW",      "HIGH", "HIGH", "MODERATE", "HIGH", "LOW",      "MODERATE"],
    # T4  (complement of T3 — swap L↔M)
    ["HIGH", "MODERATE", "HIGH", "HIGH", "LOW",      "HIGH", "MODERATE", "LOW"],
    # T5
    ["LOW",      "HIGH", "HIGH", "MODERATE", "HIGH", "LOW",      "HIGH", "MODERATE"],
    # T6  (complement of T5 — swap L↔M)
    ["MODERATE", "HIGH", "HIGH", "LOW",      "HIGH", "MODERATE", "HIGH", "LOW"],
]

# 10-block templates  (2×L + 3×M + 5×H)
# ----------------------------------------
# Six complementary pairs:  (T1,T4), (T2,T5), (T3,T6)
# No run of more than 2 consecutive identical levels.
#
#   pos:   0    1    2    3    4    5    6    7    8    9
# T1 =    L    H    M    H    M    H    H    L    H    M
# T2 =    M    H    L    H    L    H    H    M    H    M   ← swap L↔M in T1 but keep counts: 2L+3M
#      Actually T1/T2 are NOT pure L↔M swaps because the counts differ (2L+3M ≠ 3M+2L impossible).
#      So T4 is defined as T1 reversed / shifted instead.

TEMPLATES_10: list[list[str]] = [
    # T1
    ["LOW",  "HIGH", "MODERATE", "HIGH", "MODERATE",
     "HIGH", "HIGH", "LOW",      "HIGH", "MODERATE"],
    # T2
    ["MODERATE", "HIGH", "LOW",  "HIGH", "MODERATE",
     "HIGH",     "HIGH", "MODERATE", "HIGH", "LOW"],
    # T3
    ["HIGH", "LOW",      "HIGH", "HIGH", "MODERATE",
     "HIGH", "MODERATE", "HIGH", "LOW",  "MODERATE"],
    # T4  (offset complement of T1)
    ["MODERATE", "HIGH", "LOW",      "HIGH", "HIGH",
     "MODERATE", "HIGH", "MODERATE", "HIGH", "LOW"],
    # T5  (offset complement of T2)
    ["LOW",  "HIGH", "HIGH", "MODERATE", "HIGH",
     "LOW",  "HIGH", "MODERATE", "HIGH", "MODERATE"],
    # T6  (offset complement of T3)
    ["HIGH", "MODERATE", "HIGH", "LOW",      "HIGH",
     "HIGH", "MODERATE", "HIGH", "MODERATE", "LOW"],
]

_TEMPLATES: dict[int, list[list[str]]] = {
    8:  TEMPLATES_8,
    10: TEMPLATES_10,
}


# ---------------------------------------------------------------------------
# Counterbalancing helpers
# ---------------------------------------------------------------------------

def _participant_rank(participant_id: str) -> int:
    """Return 0-based rank from participant_assignments.yaml, or 0 on failure."""
    try:
        import yaml  # type: ignore[import-untyped]

        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        participants: list[str] = [
            str(entry["id"]) if isinstance(entry, dict) else str(entry)
            for entry in config.get("participants", [])
        ]
        if participant_id in participants:
            return participants.index(participant_id)
    except Exception:  # noqa: BLE001
        pass
    return 0


def assign_template(
    participant_id: str,
    condition: int,
    n_blocks: int,
) -> list[str]:
    """Return the counterbalanced level sequence for this participant/condition.

    Conditions 1 and 2 use complementary templates (offset by half the
    template list length), matching the calibration generator's scheme.
    """
    templates = _TEMPLATES[n_blocks]
    rank = _participant_rank(participant_id)
    offset = 0 if condition == 1 else len(templates) // 2
    return templates[(rank + offset) % len(templates)]


# ---------------------------------------------------------------------------
# Scenario writer
# ---------------------------------------------------------------------------

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
    """Assemble and write (or print) the full adaptive-automation scenario."""

    n_blocks = len(level_sequence)
    scenario_duration_sec = n_blocks * BLOCK_DURATION_SEC

    pid_tok = "${OPENMATB_PARTICIPANT}"
    sid_tok = "${OPENMATB_SESSION}"
    payload = f"pid={pid_tok}|sid={sid_tok}"

    cond_marker = f"adaptive_automation/{condition}"
    scene_start_m = f"STUDY/V0/{cond_marker}/START|{payload}"
    scene_end_m   = f"STUDY/V0/{cond_marker}/END|{payload}"

    # Generate all blocks
    all_lines: list[_Line] = []
    for block_index, level_label in enumerate(level_sequence):
        d = level_difficulties[level_label]
        block_start = block_index * BLOCK_DURATION_SEC
        seed_str = f"{participant_id}|adaptive|{condition}|{block_index}"
        block_rng = random.Random(seed_str)
        block_lines = generate_block_lines(
            level_label=level_label,
            d=d,
            block_index=block_index,
            block_start_sec=block_start,
            participant_id=participant_id,
            condition=condition,
            rng=block_rng,
            is_first_block=(block_index == 0),
            condition_prefix="adaptive_automation",
        )
        all_lines.extend(block_lines)

    all_lines.sort(key=lambda ln: ln.time_sec)

    end_t = _fmt_t(scenario_duration_sec)
    level_ds_str = ", ".join(
        f"{lbl}={level_difficulties[lbl]:.4f}"
        for lbl in ("LOW", "MODERATE", "HIGH")
    )
    block_order_str = " ".join(s[0] for s in level_sequence)   # e.g. "L H M H…"

    lines_out: list[str] = [
        f"# OpenMATB Scenario: {output_path.name}",
        f"# Adaptive automation -- condition {condition}, {n_blocks} min continuous",
        f"#",
        f"# Participant  : {participant_id}",
        f"# d_final      : {d_final:.4f}",
        f"# delta        : {delta:.4f}",
        f"# Level anchors: {level_ds_str}",
        f"# Block order  : {block_order_str}  (L=LOW M=MODERATE H=HIGH)",
        f"# Block split  : {level_sequence.count('HIGH')}×HIGH + "
        f"{level_sequence.count('LOW')}×LOW + "
        f"{level_sequence.count('MODERATE')}×MODERATE"
        f"  ({n_blocks} blocks total)",
        f"# Seed basis   : <participant>|adaptive|<condition>|<block_index>",
        f"# Generated by : scripts/generate_adaptive_automation_scenarios.py",
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
            "Generate a pre-scheduled adaptive-automation MATB scenario "
            "from a participant's staircase d_final.\n\n"
            "Block split: 8-block → 2×LOW + 2×MODERATE + 4×HIGH\n"
            "             10-block → 2×LOW + 3×MODERATE + 5×HIGH"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--participant", required=True,
        help="Participant ID (e.g. P001)",
    )
    parser.add_argument(
        "--condition", type=int, choices=[1, 2], required=True,
        help="Study condition (1 or 2; determines counterbalancing offset)",
    )
    parser.add_argument(
        "--n-blocks", type=int, choices=list(SUPPORTED_N_BLOCKS),
        default=DEFAULT_N_BLOCKS,
        help=f"Total number of 1-minute blocks (default: {DEFAULT_N_BLOCKS})",
    )

    d_source = parser.add_mutually_exclusive_group(required=True)
    d_source.add_argument(
        "--calibration-csv", type=Path, metavar="CSV",
        help="Path to an OpenMATB staircase-calibration session CSV",
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
        help=f"Output directory (default: experiment/scenarios/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print scenario to stdout instead of writing to disk",
    )

    args = parser.parse_args()

    # -- Resolve d_final ---------------------------------------------------
    if args.d_final is not None:
        d_final = args.d_final
        if not 0.0 <= d_final <= 1.0:
            parser.error(f"--d-final must be in [0.0, 1.0], got {d_final}")
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

    level_sequence = assign_template(args.participant, args.condition, args.n_blocks)
    block_order_str = " ".join(s[0] for s in level_sequence)
    print(f"Block order ({args.n_blocks}-block, condition {args.condition}): {block_order_str}")
    print(
        f"  HIGH={level_sequence.count('HIGH')}  "
        f"MODERATE={level_sequence.count('MODERATE')}  "
        f"LOW={level_sequence.count('LOW')}"
    )

    # -- Parameter preview -------------------------------------------------
    for label, d in level_difficulties.items():
        p = make_task_params(d)
        print(
            f"  {label:8s}  d={d:.4f}  "
            f"track_ms={int(round(p.track_update_ms))}  "
            f"joystick={int(round(p.track_joystick_force))}  "
            f"leak={p.resman_loss_a_per_min} ml/min"
        )

    filename = (
        f"adaptive_automation_{args.participant.lower()}"
        f"_c{args.condition}_{args.n_blocks}min.txt"
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
        print(
            f"\nQuick check (dry run):\n"
            f"  python scripts/generate_adaptive_automation_scenarios.py "
            f"--participant {args.participant} "
            f"--condition {args.condition} "
            f"--n-blocks {args.n_blocks} "
            f"--d-final {d_final:.4f} --dry-run"
        )


if __name__ == "__main__":
    main()
