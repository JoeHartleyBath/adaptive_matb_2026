"""Compare task KPIs (tracking RMSE, SysMon accuracy, comms accuracy) across
workload levels (LOW / MODERATE / HIGH) for up to four conditions:

  - Calibration condition 1 (--cal-c1)
  - Calibration condition 2 (--cal-c2)
  - Adaptation condition    (--adaptation)
  - Control condition       (--control)

Each argument is the path to an OpenMATB session CSV for that condition.
At least one must be supplied.  Conditions not supplied are skipped.

Workload level is parsed from LSL block markers:
    STUDY/V0/.../block_NN/LEVEL/START  and  .../END

KPIs are computed per level by merging all blocks at that level within
the condition, then calling the existing derived-KPI functions from
performance.summarise_openmatb_performance.

Usage
-----
    python scripts/analysis/performance_compare_by_level.py \\
        --cal-c1     /path/to/cal_c1.csv \\
        --cal-c2     /path/to/cal_c2.csv \\
        --adaptation /path/to/adaptation.csv \\
        --control    /path/to/control.csv \\
        --out        results/figures/P001/S001/p001_s001_performance_by_level.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from performance.summarise_openmatb_performance import (  # noqa: E402
    _collect_performance_rows,
    _compute_derived_kpis,
    _parse_float_list,
)

# ---------------------------------------------------------------------------
# Block parsing (same logic as analyse_adaptation_session.load_session_blocks)
# ---------------------------------------------------------------------------

_LEVEL_RE = re.compile(r"/block_(\d+)/(\w+)$")

_LEVELS_ORDERED = ("HIGH", "MODERATE", "LOW")


def _parse_blocks(csv_path: Path) -> list[dict]:
    """Return sorted list of dicts with keys: level, block_num, start_sec, end_sec."""
    starts: dict[str, tuple[float, str, int]] = {}
    ends: dict[str, float] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("type") or "").strip().lower() != "event":
                continue
            if (row.get("module") or "").strip().lower() != "labstreaminglayer":
                continue
            if (row.get("address") or "").strip().lower() != "marker":
                continue

            try:
                t = float(row.get("scenario_time") or "")
            except ValueError:
                continue

            raw = (row.get("value") or "").split("|", 1)[0].strip()
            body = raw[len("STUDY/V0/"):] if raw.startswith("STUDY/V0/") else raw

            if body.endswith("/START"):
                base = body[: -len("/START")]
                m = _LEVEL_RE.search(base)
                if m:
                    starts[base] = (t, m.group(2).upper(), int(m.group(1)))
            elif body.endswith("/END"):
                ends[body[: -len("/END")]] = t

    blocks = []
    for base, (start_t, level, block_num) in starts.items():
        end_t = ends.get(base)
        if end_t is None or end_t <= start_t:
            continue
        blocks.append({
            "level": level,
            "block_num": block_num,
            "start_sec": start_t,
            "end_sec": end_t,
        })

    blocks.sort(key=lambda b: b["start_sec"])
    return blocks


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------

def _compute_rt_kpis(by_module: dict) -> dict:
    """Mean response time (s) for sysmon and communications.

    NaN response_time (= no response / MISS) is excluded by _parse_float_list,
    so the mean reflects only trials where the participant actually responded.
    """
    import math

    result: dict[str, dict] = {}
    for task in ("sysmon", "communications"):
        mod = by_module.get(task)
        if not isinstance(mod, dict):
            continue
        rt_raw = mod.get("response_time")
        if not isinstance(rt_raw, list):
            continue
        rts = _parse_float_list(rt_raw)
        if not rts:
            continue
        mean_rt = sum(rts) / len(rts)
        std_rt = math.sqrt(sum((v - mean_rt) ** 2 for v in rts) / len(rts))
        result[task] = {
            "rt_mean_s": round(mean_rt, 3),
            "rt_std_s":  round(std_rt,  3),
            "rt_n":      len(rts),
        }
    return result


def _kpis_for_condition(csv_path: Path, blocks: list[dict]) -> dict[str, dict]:
    """Compute merged KPIs per level for one condition's session CSV."""
    by_level: dict[str, dict[str, dict[str, list[str]]]] = {}

    for block in blocks:
        level = block["level"]
        window = (block["start_sec"], block["end_sec"])
        row_data = _collect_performance_rows(csv_path, window=window)
        merged = by_level.setdefault(level, {})
        for mod, metrics in row_data.items():
            mod_entry = merged.setdefault(mod, {})
            for metric, vals in metrics.items():
                mod_entry.setdefault(metric, []).extend(vals)

    result: dict[str, dict] = {}
    for level, mod_data in by_level.items():
        kpis = _compute_derived_kpis(mod_data)
        for task, rt_stats in _compute_rt_kpis(mod_data).items():
            kpis.setdefault(task, {}).update(rt_stats)
        result[level] = kpis
    return result


def _block_summary(blocks: list[dict], level: str) -> dict:
    lvl_blocks = [b for b in blocks if b["level"] == level]
    total_s = sum(b["end_sec"] - b["start_sec"] for b in lvl_blocks)
    return {"n_blocks": len(lvl_blocks), "total_s": round(total_s, 1)}


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_COL_W = 13


def _fmt_rmse(kpis: dict) -> str:
    v = kpis.get("tracking", {}).get("center_deviation_rmse")
    return f"{v:.4f}" if v is not None else "\u2014"


def _fmt_acc(kpis: dict, task: str) -> str:
    v = kpis.get(task, {}).get("accuracy")
    return f"{v:.1%}" if v is not None else "\u2014"


def _fmt_rt(kpis: dict, task: str) -> str:
    v = kpis.get(task, {}).get("rt_mean_s")
    return f"{v / 1000:.2f}s" if v is not None else "\u2014"


def _print_table(conditions: dict[str, dict]) -> None:
    """Print a console summary table.

    conditions: { condition_label: { level: { kpis, n_blocks, total_s } } }
    """
    header = (
        f"{'Condition':<16} {'Level':<10} "
        f"{'Track RMSE':>{_COL_W}} {'SysMon Acc':>{_COL_W}} {'SysMon RT':>{_COL_W}} "
        f"{'Comms Acc':>{_COL_W}} {'Comms RT':>{_COL_W}} "
        f"{'N blocks':>9} {'Total s':>9}"
    )
    print("\n" + header)
    print("-" * len(header))

    for cond_label, level_data in conditions.items():
        for level in _LEVELS_ORDERED:
            entry = level_data.get(level)
            if entry is None:
                continue
            kpis = entry["kpis"]
            print(
                f"{cond_label:<16} {level:<10} "
                f"{_fmt_rmse(kpis):>{_COL_W}} "
                f"{_fmt_acc(kpis, 'sysmon'):>{_COL_W}} "
                f"{_fmt_rt(kpis, 'sysmon'):>{_COL_W}} "
                f"{_fmt_acc(kpis, 'communications'):>{_COL_W}} "
                f"{_fmt_rt(kpis, 'communications'):>{_COL_W}} "
                f"{entry['n_blocks']:>9} "
                f"{entry['total_s']:>9.1f}"
            )
        print()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# X-axis goes easy → hard (left to right)
_LEVELS_X_ORDER  = ("LOW", "MODERATE", "HIGH")
_LEVELS_X_LABELS = ("LOW", "MOD", "HIGH")

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


def _plot_difficulty_manipulation(
    conditions: dict[str, dict],
    plot_path: Path | None,
) -> None:
    """2-row × 3-column figure checking that KPIs respond to difficulty.

    Row 0 — accuracy / error metrics:
      [0] Tracking RMSE  (should increase LOW→HIGH)
      [1] SysMon accuracy (should decrease LOW→HIGH)
      [2] Comms accuracy  (should decrease LOW→HIGH)

    Row 1 — response time:
      [3] SysMon RT  (should increase LOW→HIGH)
      [4] Comms RT   (should increase LOW→HIGH)
      [5] hidden

    Pass plot_path to save the figure; omit (None) to show interactively.
    """
    import matplotlib.pyplot as plt

    x_pos   = {level: i for i, level in enumerate(_LEVELS_X_ORDER)}
    x_ticks = list(range(len(_LEVELS_X_ORDER)))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), sharey=False)
    fig.suptitle(
        "Difficulty manipulation check: task KPIs by workload level",
        fontsize=12,
        fontweight="bold",
    )

    # (flat_index, title, extractor, y_label, directional_note, is_pct)
    kpi_specs = [
        (
            0, "Tracking RMSE",
            lambda k: k.get("tracking", {}).get("center_deviation_rmse"),
            "Centre deviation RMSE",
            "should increase ↑",
            False,
        ),
        (
            1, "SysMon accuracy",
            lambda k: k.get("sysmon", {}).get("accuracy"),
            "Accuracy",
            "should decrease ↓",
            True,
        ),
        (
            2, "Comms accuracy",
            lambda k: k.get("communications", {}).get("accuracy"),
            "Accuracy",
            "should decrease ↓",
            True,
        ),
        (
            3, "SysMon response time",
            lambda k: (rt / 1000) if (rt := k.get("sysmon", {}).get("rt_mean_s")) is not None else None,
            "Mean RT (s)",
            "should increase \u2191",
            False,
        ),
        (
            4, "Comms response time",
            lambda k: (rt / 1000) if (rt := k.get("communications", {}).get("rt_mean_s")) is not None else None,
            "Mean RT (s)",
            "should increase ↑",
            False,
        ),
    ]

    flat_axes = axes.flat

    for ax_i, title, extractor, y_label, note, is_pct in kpi_specs:
        ax = flat_axes[ax_i]
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(_LEVELS_X_LABELS)
        ax.set_xlabel("Workload level", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)

        if is_pct:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0%}")
            )
            ax.set_ylim(0, 1.08)

        plotted_any = False
        for cond_label, level_data in conditions.items():
            xs: list[int] = []
            ys: list[float] = []
            for level in _LEVELS_X_ORDER:
                entry = level_data.get(level)
                if entry is None:
                    continue
                v = extractor(entry["kpis"])
                if v is not None:
                    xs.append(x_pos[level])
                    ys.append(float(v))

            if not xs:
                continue

            ax.plot(
                xs, ys,
                marker="o", linewidth=2, markersize=7,
                color=_COND_COLOURS.get(cond_label),
                label=_COND_DISPLAY.get(cond_label, cond_label),
            )
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=9)

        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused bottom-right panel
    flat_axes[5].set_visible(False)

    # Single shared legend below all panels
    handles: list = []
    labels:  list = []
    for ax in flat_axes:
        if not ax.get_visible():
            continue
        h, l = ax.get_legend_handles_labels()
        for handle, lbl in zip(h, l):
            if lbl not in labels:
                handles.append(handle)
                labels.append(lbl)

    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=len(handles),
            frameon=False, fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

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
        description="Compare task KPIs by workload level across conditions.",
    )
    parser.add_argument("--cal-c1",     type=Path, default=None,
                        help="Calibration condition 1 session CSV")
    parser.add_argument("--cal-c2",     type=Path, default=None,
                        help="Calibration condition 2 session CSV")
    parser.add_argument("--adaptation", type=Path, default=None,
                        help="Adaptation condition session CSV")
    parser.add_argument("--control",    type=Path, default=None,
                        help="Control condition session CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (optional)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save figure to this path (e.g. results/figures/P001/S001/p001_s001_fig03_kpi_by_level.png). "
                             "Omit to show interactively.")
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

    conditions: dict[str, dict] = {}

    for label, csv_path in active:
        print(f"Processing {label}: {csv_path.name}")
        blocks = _parse_blocks(csv_path)
        if not blocks:
            print(f"  WARNING: No block markers found — skipping {label}")
            continue

        level_counts: dict[str, int] = {}
        for b in blocks:
            level_counts[b["level"]] = level_counts.get(b["level"], 0) + 1
        print(f"  {len(blocks)} blocks: {level_counts}")

        kpis_by_level = _kpis_for_condition(csv_path, blocks)

        conditions[label] = {
            level: {
                "kpis": kpis_by_level.get(level, {}),
                **_block_summary(blocks, level),
            }
            for level in _LEVELS_ORDERED
            if any(b["level"] == level for b in blocks)
        }

    _print_table(conditions)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(conditions, indent=2))
        print(f"Results written to {args.out}")

    _plot_difficulty_manipulation(conditions, args.plot)


if __name__ == "__main__":
    main()
