"""Plot MWL adaptation session timeline.

Reads the adaptation audit CSV (and optionally the OpenMATB session CSV for
block boundaries) and produces a timeline figure showing:
  - Raw P(overload) as light-grey trace
  - EMA-smoothed MWL as dark trace
  - Decision threshold as red dashed line
  - Assistance-ON periods shaded green
  - Workload block backgrounds colour-coded by level (if session CSV given)

Output follows the naming convention:
    results/figures/adaptation__fig01__mwl_timeline.png

Usage
-----
    python scripts/plot_adaptation_session.py \
        --audit   /path/to/adaptation_audit.csv \
        --session /path/to/openmatb_session.csv \
        --out     results/figures/adaptation__fig01__mwl_timeline.png
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptation.adaptation_logger import COLUMNS as AUDIT_COLUMNS  # noqa: E402

# ---------------------------------------------------------------------------
# Data structures (same as analyse_adaptation_session.py)
# ---------------------------------------------------------------------------

@dataclass
class AuditRow:
    timestamp_lsl: float
    scenario_time_s: float
    mwl_raw: float
    mwl_smoothed: float
    signal_quality: float
    threshold: float
    action: str
    assistance_on: bool
    cooldown_remaining_s: float
    hold_counter_s: float
    reason: str


@dataclass
class BlockSegment:
    name: str
    level: str
    block_num: int
    start_sec: float
    end_sec: float


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_audit_csv(path: Path) -> list[AuditRow]:
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


_LEVEL_RE = re.compile(r"/block_(\d+)/(\w+)$")


def _load_session_blocks(csv_path: Path) -> list[BlockSegment]:
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
                    starts[base] = (t, m.group(2).upper(), int(m.group(1)))
            elif body.endswith("/END"):
                base = body[:-len("/END")]
                ends[base] = t

    segments: list[BlockSegment] = []
    for base, (start_t, level, block_num) in starts.items():
        end_t = ends.get(base)
        if end_t is None or end_t <= start_t:
            continue
        segments.append(BlockSegment(
            name=base, level=level, block_num=block_num,
            start_sec=start_t, end_sec=end_t,
        ))
    segments.sort(key=lambda s: s.start_sec)
    return segments


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

_LEVEL_COLOURS = {
    "HIGH":     "tab:red",
    "MODERATE": "tab:orange",
    "LOW":      "tab:blue",
}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_timeline(
    audit_rows: list[AuditRow],
    blocks: list[BlockSegment] | None,
    out_path: Path,
) -> None:
    """Single-panel timeline: raw + smoothed MWL, threshold, assist shading."""
    t = np.array([r.scenario_time_s for r in audit_rows])
    raw = np.array([r.mwl_raw for r in audit_rows])
    smoothed = np.array([r.mwl_smoothed for r in audit_rows])
    threshold = np.array([r.threshold for r in audit_rows])
    assist_on = np.array([r.assistance_on for r in audit_rows])

    fig, ax = plt.subplots(figsize=(14, 5))

    # Block backgrounds
    if blocks:
        for blk in blocks:
            colour = _LEVEL_COLOURS.get(blk.level, "grey")
            ax.axvspan(blk.start_sec, blk.end_sec,
                       color=colour, alpha=0.07, zorder=0)
            ax.text(
                (blk.start_sec + blk.end_sec) / 2,
                1.02, blk.level,
                ha="center", va="bottom",
                fontsize=6, color=colour, alpha=0.8,
                transform=ax.get_xaxis_transform(),
            )

    # Assistance-ON shading
    _shade_assist(ax, t, assist_on)

    # Raw MWL
    ax.plot(t, raw, color="0.65", alpha=0.5, linewidth=0.5, label="raw P(overload)")

    # Smoothed MWL
    ax.plot(t, smoothed, color="0.2", linewidth=1.0, label="smoothed")

    # Threshold
    ax.plot(t, threshold, color="red", linestyle="--", linewidth=0.8,
            label="threshold", alpha=0.7)

    ax.set_xlabel("Scenario time (s)", fontsize=9)
    ax.set_ylabel("MWL  P(overload)", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper right")

    # Assist-ON legend entry (green patch)
    from matplotlib.patches import Patch
    legend_handles = ax.get_legend().legend_handles
    legend_handles.append(Patch(facecolor="green", alpha=0.15, label="assist ON"))
    ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

    duration_min = (t[-1] - t[0]) / 60
    n_on = int(assist_on.sum())
    pct_on = 100.0 * n_on / len(assist_on)
    ax.set_title(
        f"MWL Adaptation Timeline  —  {duration_min:.1f} min  |  "
        f"assist ON {pct_on:.0f}% ({n_on}/{len(assist_on)} ticks)",
        fontsize=10,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _shade_assist(ax, t: np.ndarray, assist_on: np.ndarray) -> None:
    """Shade contiguous assist-ON regions."""
    if len(t) < 2:
        return
    dt = np.median(np.diff(t))
    in_region = False
    start = 0.0
    for i in range(len(t)):
        if assist_on[i] and not in_region:
            start = t[i] - dt / 2
            in_region = True
        elif not assist_on[i] and in_region:
            end = t[i - 1] + dt / 2
            ax.axvspan(start, end, color="green", alpha=0.15, zorder=1)
            in_region = False
    if in_region:
        ax.axvspan(start, t[-1] + dt / 2, color="green", alpha=0.15, zorder=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot MWL adaptation session timeline.",
    )
    parser.add_argument("--audit", type=Path, required=True,
                        help="Path to adaptation_audit.csv")
    parser.add_argument("--session", type=Path, default=None,
                        help="Path to OpenMATB session CSV (for block boundaries)")
    parser.add_argument("--out", type=Path,
                        default=_REPO_ROOT / "results" / "figures"
                        / "adaptation__fig01__mwl_timeline.png",
                        help="Output figure path")
    args = parser.parse_args()

    print("Loading audit CSV...")
    audit_rows = _load_audit_csv(args.audit)
    print(f"  {len(audit_rows)} ticks "
          f"({audit_rows[0].scenario_time_s:.1f}s – "
          f"{audit_rows[-1].scenario_time_s:.1f}s)")

    blocks = None
    if args.session:
        print("Loading session blocks...")
        blocks = _load_session_blocks(args.session)
        if blocks:
            levels = {}
            for b in blocks:
                levels[b.level] = levels.get(b.level, 0) + 1
            print(f"  {len(blocks)} blocks: {levels}")
        else:
            print("  WARNING: No block markers found.")

    plot_timeline(audit_rows, blocks, args.out)


if __name__ == "__main__":
    main()
