"""Compare NASA-TLX subscale ratings across study conditions.

Extracts TLX data from OpenMATB session CSVs, where responses are logged as
    type=performance, module=genericscales, address=<subscale>, value=<score>
within the marker window STUDY/V0/TLX/{condition}/START|END.

Conditions are supplied via named arguments; any subset may be provided.
Raw TLX (unweighted mean of 6 subscales) is computed automatically.

Usage
-----
    python scripts/analysis/tlx_compare_conditions.py \\
        --cal-c1     /path/to/cal_c1.csv \\
        --cal-c2     /path/to/cal_c2.csv \\
        --adaptation /path/to/adaptation.csv \\
        --control    /path/to/control.csv \\
        --out        results/figures/P001/S001/p001_s001_tlx_comparison.json \\
        --plot       results/figures/P001/S001/p001_s001_fig02_tlx.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical subscale names as stored by the genericscales plugin (lowercase).
_SUBSCALES = (
    "mental demand",
    "physical demand",
    "time pressure",
    "effort",
    "frustration",
    "performance",
)

_SUBSCALE_LABELS = {
    "mental demand":   "Mental\nDemand",
    "physical demand": "Physical\nDemand",
    "time pressure":   "Time\nPressure",
    "effort":          "Effort",
    "frustration":     "Frustration",
    "performance":     "Performance",
}

_COND_COLOURS = {
    "cal_c1":     "#4C72B0",
    "cal_c2":     "#DD8452",
    "adaptation": "#55A868",
    "control":    "#C44E52",
}
_COND_DISPLAY = {
    "cal_c1":     "Cal C1",
    "cal_c2":     "Cal C2",
    "adaptation": "Adaptation",
    "control":    "Control",
}
_COND_ORDER = ("cal_c1", "cal_c2", "adaptation", "control")

# Scale maximum (OpenMATB genericscales uses 0–10 by default)
_SCALE_MAX = 10.0


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _find_tlx_window(csv_path: Path) -> tuple[float, float] | None:
    """Return (start_sec, end_sec) of the TLX marker window, or None."""
    start_t: float | None = None
    end_t:   float | None = None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("type") or "").strip().lower() != "event":
                continue
            if (row.get("module") or "").strip().lower() != "labstreaminglayer":
                continue
            if (row.get("address") or "").strip().lower() != "marker":
                continue

            raw = (row.get("value") or "").split("|", 1)[0].strip()
            if "TLX" not in raw:
                continue

            try:
                t = float(row.get("scenario_time") or "")
            except ValueError:
                continue

            if raw.endswith("/START") and start_t is None:
                start_t = t
            elif raw.endswith("/END") and end_t is None:
                end_t = t

    if start_t is not None and end_t is not None:
        return start_t, end_t
    return None


def _extract_tlx(csv_path: Path) -> dict[str, float] | None:
    """Return dict of subscale -> score for the TLX in this session CSV.

    Uses the TLX marker window when present; falls back to all genericscales
    performance rows (robust to scenarios without explicit TLX markers).
    Returns None if no TLX data found.
    """
    window = _find_tlx_window(csv_path)

    scores: dict[str, float] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("type") or "").strip().lower() != "performance":
                continue
            if (row.get("module") or "").strip().lower() != "genericscales":
                continue

            try:
                t = float(row.get("scenario_time") or "")
            except ValueError:
                continue

            if window is not None and not (window[0] <= t <= window[1]):
                continue

            subscale = (row.get("address") or "").strip().lower()
            raw_val  = (row.get("value")   or "").strip()
            try:
                scores[subscale] = float(raw_val)
            except ValueError:
                continue

    return scores if scores else None


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_LABEL_W = 18
_COL_W   = 10


def _print_table(results: dict[str, dict]) -> None:
    """Print subscale scores and raw TLX mean per condition."""
    cond_labels = [c for c in _COND_ORDER if c in results]

    header = f"{'Subscale':<{_LABEL_W}}" + "".join(
        f"{_COND_DISPLAY[c]:>{_COL_W}}" for c in cond_labels
    )
    print("\n" + header)
    print("-" * len(header))

    for sub in _SUBSCALES:
        row_str = f"{sub.title():<{_LABEL_W}}"
        for c in cond_labels:
            v = results[c].get(sub)
            row_str += f"{v:>{_COL_W}.2f}" if v is not None else f"{'—':>{_COL_W}}"
        print(row_str)

    print("-" * len(header))
    row_str = f"{'Raw TLX (mean)':<{_LABEL_W}}"
    for c in cond_labels:
        v = results[c].get("raw_tlx")
        row_str += f"{v:>{_COL_W}.2f}" if v is not None else f"{'—':>{_COL_W}}"
    print(row_str)
    print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_tlx(results: dict[str, dict], plot_path: Path | None) -> None:
    """Grouped bar chart: subscales on x-axis, one bar group per condition.

    A separate summary panel shows the raw TLX mean per condition.
    """
    import math
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    cond_labels = [c for c in _COND_ORDER if c in results]
    n_conds = len(cond_labels)
    n_subs  = len(_SUBSCALES)

    x = np.arange(n_subs)
    bar_w = 0.7 / max(n_conds, 1)

    fig = plt.figure(figsize=(11, 5.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[4.5, 1.2], figure=fig,
                            wspace=0.08)
    ax_bars    = fig.add_subplot(gs[0])
    ax_summary = fig.add_subplot(gs[1])

    fig.suptitle("NASA-TLX by condition", fontsize=12, fontweight="bold")

    # --- Subscale bars ---
    for i, cond in enumerate(cond_labels):
        offsets = (i - (n_conds - 1) / 2) * bar_w
        vals = [results[cond].get(sub) for sub in _SUBSCALES]
        ys   = [v if v is not None else 0.0 for v in vals]
        mask = [v is None for v in vals]

        bars = ax_bars.bar(
            x + offsets, ys,
            width=bar_w * 0.9,
            color=_COND_COLOURS.get(cond),
            label=_COND_DISPLAY.get(cond, cond),
            alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        # Cross-hatch bars where data is missing
        for bar, missing in zip(bars, mask):
            if missing:
                bar.set_hatch("//")
                bar.set_alpha(0.3)

        # Value labels above bars
        for bar, missing in zip(bars, mask):
            if not missing:
                h = bar.get_height()
                ax_bars.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.08,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6.5,
                )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(
        [_SUBSCALE_LABELS[s] for s in _SUBSCALES], fontsize=9
    )
    ax_bars.set_ylim(0, _SCALE_MAX * 1.18)
    ax_bars.set_ylabel("Rating (0–10)", fontsize=9)
    ax_bars.set_xlabel("Subscale", fontsize=9)
    ax_bars.axhline(5, color="grey", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_bars.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.legend(fontsize=8, frameon=False, loc="upper right")

    # --- Raw TLX summary panel ---
    raw_vals  = [results[c].get("raw_tlx") for c in cond_labels]
    colours   = [_COND_COLOURS.get(c, "#888888") for c in cond_labels]
    disp_lbls = [_COND_DISPLAY.get(c, c) for c in cond_labels]

    valid = [(lbl, v, col) for lbl, v, col in zip(disp_lbls, raw_vals, colours)
             if v is not None]

    if valid:
        lbls, vals_raw, cols = zip(*valid)
        y_pos = np.arange(len(vals_raw))
        ax_summary.barh(
            y_pos, vals_raw,
            color=list(cols), alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )
        for yi, v in zip(y_pos, vals_raw):
            ax_summary.text(
                v + 0.1, yi, f"{v:.2f}",
                va="center", fontsize=8,
            )
        ax_summary.set_yticks(list(y_pos))
        ax_summary.set_yticklabels(list(lbls), fontsize=9)
        ax_summary.set_xlim(0, _SCALE_MAX * 1.2)
        ax_summary.set_ylim(-0.6, len(valid) - 0.4)
        ax_summary.axvline(5, color="grey", linewidth=0.8,
                           linestyle="--", alpha=0.4)
        ax_summary.set_xlabel("Raw TLX mean", fontsize=9)
        ax_summary.set_title("Raw TLX", fontsize=9)
        ax_summary.spines["top"].set_visible(False)
        ax_summary.spines["right"].set_visible(False)
        ax_summary.grid(axis="x", linestyle="--", alpha=0.25)
    else:
        ax_summary.text(0.5, 0.5, "No data",
                        ha="center", va="center", color="grey",
                        transform=ax_summary.transAxes)
        ax_summary.axis("off")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.14, wspace=0.12)

    if plot_path is not None:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare NASA-TLX subscale ratings across conditions.",
    )
    parser.add_argument("--cal-c1",     type=Path, default=None,
                        help="Calibration condition 1 session CSV")
    parser.add_argument("--cal-c2",     type=Path, default=None,
                        help="Calibration condition 2 session CSV")
    parser.add_argument("--adaptation", type=Path, default=None,
                        help="Adaptation condition session CSV")
    parser.add_argument("--control",    type=Path, default=None,
                        help="Control condition session CSV")
    parser.add_argument("--out",  type=Path, default=None,
                        help="Output JSON path (optional)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save figure to this path. Omit to show interactively.")
    args = parser.parse_args()

    named_inputs: list[tuple[str, Path]] = [
        ("cal_c1",     args.cal_c1),
        ("cal_c2",     args.cal_c2),
        ("adaptation", args.adaptation),
        ("control",    args.control),
    ]
    active = [(label, p) for label, p in named_inputs if p is not None]
    if not active:
        sys.exit(
            "ERROR: Supply at least one of "
            "--cal-c1 / --cal-c2 / --adaptation / --control"
        )

    for label, p in active:
        if not p.exists():
            sys.exit(f"ERROR: File not found for {label}: {p}")

    results: dict[str, dict] = {}

    for label, csv_path in active:
        print(f"Processing {label}: {csv_path.name}")
        scores = _extract_tlx(csv_path)
        if scores is None:
            print(f"  WARNING: No TLX data found — skipping {label}")
            continue

        # Raw TLX = unweighted mean of the six canonical subscales
        subscale_vals = [scores[s] for s in _SUBSCALES if s in scores]
        raw_tlx = sum(subscale_vals) / len(subscale_vals) if subscale_vals else None

        entry = {s: scores.get(s) for s in _SUBSCALES}
        entry["raw_tlx"] = raw_tlx

        for sub in _SUBSCALES:
            if sub in scores:
                print(f"  {sub.title():<18} {scores[sub]:.2f}")
        if raw_tlx is not None:
            print(f"  {'Raw TLX':<18} {raw_tlx:.2f}")

        results[label] = entry

    if not results:
        sys.exit("No TLX data found in any of the supplied files.")

    _print_table(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.out}")

    _plot_tlx(results, args.plot)


if __name__ == "__main__":
    main()
