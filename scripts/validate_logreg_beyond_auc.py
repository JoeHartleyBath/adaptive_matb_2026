"""Temporal dynamics validation plots for the LogReg MWL classifier.

Reads the per-epoch prediction CSV from extract_logreg_epoch_predictions.py
and produces:
  1. Group-averaged P(high) traces by condition (high-cog vs low-cog)
  2. Per-participant individual trace panels (small multiples)

Primary evidence uses y_prob_group (cross-subject model, no data leakage).
Personalised model shown as secondary overlay.

Usage:
    python scripts/validate_logreg_beyond_auc.py
    python scripts/validate_logreg_beyond_auc.py --csv results/test_pretrain/logreg_epoch_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT    = Path(__file__).resolve().parent.parent
_DEFAULT_CSV  = _REPO_ROOT / "results" / "test_pretrain" / "logreg_epoch_predictions.csv"
_DEFAULT_JSON = _REPO_ROOT / "results" / "test_pretrain" / "logreg_fold_coefficients.json"
_FIG_DIR      = _REPO_ROOT / "results" / "figures"
_STEP_S       = 0.5  # epoch step size in seconds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(csv_path: Path) -> dict:
    """Load CSV into a structured dict keyed by pid."""
    data: dict[str, list[dict]] = {}
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            row = dict(zip(header, vals))
            row["block_idx"]         = int(row["block_idx"])
            row["epoch_within_block"] = int(row["epoch_within_block"])
            row["epoch_idx"]         = int(row["epoch_idx"])
            row["y_true"]            = int(row["y_true"])
            row["y_prob_group"]      = float(row["y_prob_group"])
            row["y_prob_personal"]   = float(row["y_prob_personal"])
            data.setdefault(row["pid"], []).append(row)
    return data


def _build_block_traces(
    rows: list[dict],
) -> list[dict]:
    """Build per-block trace data for one participant.

    Returns list of dicts with keys: block_idx, y_true, n_epochs,
    time_s, prob_group, prob_personal.
    """
    blocks_by_idx: dict[int, list[dict]] = {}
    for r in rows:
        blocks_by_idx.setdefault(r["block_idx"], []).append(r)

    traces = []
    for bidx in sorted(blocks_by_idx.keys()):
        block_rows = sorted(blocks_by_idx[bidx], key=lambda r: r["epoch_within_block"])
        n = len(block_rows)
        traces.append({
            "block_idx":     bidx,
            "y_true":        block_rows[0]["y_true"],
            "n_epochs":      n,
            "time_s":        np.arange(n) * _STEP_S,
            "prob_group":    np.array([r["y_prob_group"] for r in block_rows]),
            "prob_personal": np.array([r["y_prob_personal"] for r in block_rows]),
        })
    return traces


# ---------------------------------------------------------------------------
# Plot 1: Group-averaged traces by condition
# ---------------------------------------------------------------------------

def plot_group_averaged_traces(
    data: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Mean ± 95% CI of P(high) across participants, by condition."""
    pids = sorted(data.keys())

    # Collect per-participant block traces, grouped by label
    high_traces: list[np.ndarray] = []
    low_traces:  list[np.ndarray] = []
    high_traces_p: list[np.ndarray] = []
    low_traces_p:  list[np.ndarray] = []

    # Find minimum block length to align traces
    min_len_high = None
    min_len_low  = None
    for pid in pids:
        for trace in _build_block_traces(data[pid]):
            if trace["y_true"] == 1:
                if min_len_high is None or trace["n_epochs"] < min_len_high:
                    min_len_high = trace["n_epochs"]
            else:
                if min_len_low is None or trace["n_epochs"] < min_len_low:
                    min_len_low = trace["n_epochs"]

    if min_len_high is None or min_len_low is None:
        print("WARNING: Missing high or low blocks, skipping group plot.")
        return

    # Collect aligned traces (average across blocks within participant first)
    for pid in pids:
        traces = _build_block_traces(data[pid])
        pid_high_g = []
        pid_high_p = []
        pid_low_g  = []
        pid_low_p  = []
        for t in traces:
            if t["y_true"] == 1:
                pid_high_g.append(t["prob_group"][:min_len_high])
                pid_high_p.append(t["prob_personal"][:min_len_high])
            else:
                pid_low_g.append(t["prob_group"][:min_len_low])
                pid_low_p.append(t["prob_personal"][:min_len_low])

        if pid_high_g:
            high_traces.append(np.mean(pid_high_g, axis=0))
            high_traces_p.append(np.mean(pid_high_p, axis=0))
        if pid_low_g:
            low_traces.append(np.mean(pid_low_g, axis=0))
            low_traces_p.append(np.mean(pid_low_p, axis=0))

    high_arr   = np.array(high_traces)     # (n_pids, n_epochs)
    low_arr    = np.array(low_traces)
    high_arr_p = np.array(high_traces_p)
    low_arr_p  = np.array(low_traces_p)

    time_high = np.arange(min_len_high) * _STEP_S
    time_low  = np.arange(min_len_low)  * _STEP_S

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, label, title in [
        (axes[0], "Group model (cross-subject)", "Group Model"),
        (axes[1], "Personalised model (WS-weak)", "Personalised Model"),
    ]:
        if "Group" in title:
            h_arr, l_arr = high_arr, low_arr
        else:
            h_arr, l_arr = high_arr_p, low_arr_p

        h_mean = np.mean(h_arr, axis=0)
        h_se   = np.std(h_arr, axis=0) / np.sqrt(h_arr.shape[0])
        l_mean = np.mean(l_arr, axis=0)
        l_se   = np.std(l_arr, axis=0) / np.sqrt(l_arr.shape[0])

        ax.plot(time_high, h_mean, color="#d32f2f", linewidth=1.5, label="High-cog blocks")
        ax.fill_between(time_high, h_mean - 1.96 * h_se, h_mean + 1.96 * h_se,
                         color="#d32f2f", alpha=0.15)
        ax.plot(time_low, l_mean, color="#1976d2", linewidth=1.5, label="Low-cog blocks")
        ax.fill_between(time_low, l_mean - 1.96 * l_se, l_mean + 1.96 * l_se,
                         color="#1976d2", alpha=0.15)

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Time within block (s)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("P(high workload)")
    fig.suptitle(
        f"Temporal Prediction Traces — Mean ± 95% CI (n = {len(pids)})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 2: Per-participant small multiples
# ---------------------------------------------------------------------------

def plot_individual_traces(
    data: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Per-participant temporal traces (group model only)."""
    pids = sorted(data.keys())
    n = len(pids)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.2),
                              sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for idx, pid in enumerate(pids):
        ax = axes_flat[idx]
        traces = _build_block_traces(data[pid])

        # Plot concatenated timeline with block boundaries
        x_offset = 0.0
        boundaries = []
        for t in traces:
            time = t["time_s"] + x_offset
            color = "#d32f2f" if t["y_true"] == 1 else "#1976d2"
            ax.plot(time, t["prob_group"], color=color, linewidth=0.8, alpha=0.8)
            boundaries.append(x_offset)
            x_offset = time[-1] + _STEP_S

        # Block boundary lines
        for bx in boundaries[1:]:
            ax.axvline(bx, color="grey", linestyle=":", linewidth=0.5, alpha=0.6)

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_title(pid, fontsize=8, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Shared labels
    fig.text(0.5, 0.01, "Time (s, concatenated blocks)", ha="center", fontsize=10)
    fig.text(0.01, 0.5, "P(high workload) — group model", va="center",
             rotation="vertical", fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="#d32f2f", linewidth=2, label="High-cog"),
        Line2D([0], [0], color="#1976d2", linewidth=2, label="Low-cog"),
    ]
    fig.legend(handles=legend_lines, loc="upper right", fontsize=9,
               framealpha=0.9, ncol=2)

    fig.suptitle(
        f"Per-Participant Prediction Traces — Group Model (n = {n})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 3: Block-transition summary (bar plot)
# ---------------------------------------------------------------------------

def plot_block_means(
    data: dict[str, list[dict]],
    out_path: Path,
) -> None:
    """Mean P(high) by condition, accounting for counterbalanced block order."""
    pids = sorted(data.keys())

    # Collect per-participant mean P(high) for each condition
    high_means: list[float] = []
    low_means:  list[float] = []
    for pid in pids:
        pid_high = []
        pid_low  = []
        for trace in _build_block_traces(data[pid]):
            block_mean = float(np.mean(trace["prob_group"]))
            if trace["y_true"] == 1:
                pid_high.append(block_mean)
            else:
                pid_low.append(block_mean)
        if pid_high:
            high_means.append(np.mean(pid_high))
        if pid_low:
            low_means.append(np.mean(pid_low))

    high_arr = np.array(high_means)
    low_arr  = np.array(low_means)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    x = np.array([0, 1])
    means = [np.mean(low_arr), np.mean(high_arr)]
    sems  = [np.std(low_arr) / np.sqrt(len(low_arr)),
             np.std(high_arr) / np.sqrt(len(high_arr))]
    colors = ["#1976d2", "#d32f2f"]

    ax.bar(x, means, yerr=[1.96 * s for s in sems], color=colors,
           edgecolor="white", linewidth=0.8, capsize=5, alpha=0.85, width=0.5)

    # Overlay individual participant dots
    jitter_low  = np.random.default_rng(0).uniform(-0.08, 0.08, len(low_arr))
    jitter_high = np.random.default_rng(1).uniform(-0.08, 0.08, len(high_arr))
    ax.scatter(np.zeros(len(low_arr)) + jitter_low, low_arr,
               color="#1976d2", alpha=0.4, s=15, edgecolors="none", zorder=3)
    ax.scatter(np.ones(len(high_arr)) + jitter_high, high_arr,
               color="#d32f2f", alpha=0.4, s=15, edgecolors="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Low-cog\nconditions", "High-cog\nconditions"], fontsize=10)
    ax.set_ylabel("Mean P(high workload) — group model")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"Condition-Level Mean Predictions (n = {len(pids)})",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Coefficient loading
# ---------------------------------------------------------------------------

def load_coefficients(json_path: Path) -> dict:
    """Load per-fold coefficients from the extraction JSON."""
    with open(json_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature name parsing helpers
# ---------------------------------------------------------------------------

# Region prefix → display name
_REGION_MAP = {
    "FM":  "Frontal Mid",
    "Fro": "Frontal",
    "FAA": "Frontal",
    "Par": "Parietal",
    "Cen": "Central",
    "Occ": "Occipital",
}

# Feature suffix → category
_CATEGORY_MAP = {
    "Theta":       "Bandpower",
    "Alpha":       "Bandpower",
    "Delta":       "Bandpower",
    "Beta":        "Bandpower",
    "Theta_Alpha": "Ratio",
    "Theta_Beta":  "Ratio",
    "Engagement":  "Ratio",
    "HjAct":       "Hjorth",
    "HjMob":       "Hjorth",
    "HjComp":      "Hjorth",
    "SpEnt":       "Entropy",
    "PeEnt":       "Entropy",
    "Skew":        "Stats",
    "Kurt":        "Stats",
    "ZCR":         "Stats",
    "1fSlope":     "Aperiodic",
}


def _parse_feature(name: str) -> tuple[str, str]:
    """Parse a feature name into (region, category)."""
    if name.startswith("wPLI"):
        return "Connectivity", "wPLI"
    if name == "FAA":
        return "Frontal", "Ratio"
    parts = name.split("_", 1)
    if len(parts) == 2:
        prefix, suffix = parts
        region = _REGION_MAP.get(prefix, prefix)
        category = _CATEGORY_MAP.get(suffix, suffix)
        return region, category
    return "Other", name


# ---------------------------------------------------------------------------
# Plot 4: Signed coefficient bar plot (cross-fold mean ± SD)
# ---------------------------------------------------------------------------

def plot_coefficient_bar(
    coef_data: dict,
    out_path: Path,
) -> None:
    """Top-K features by mean absolute group-model weight across LOSO folds."""
    folds = coef_data["folds"]
    pids = sorted(folds.keys())

    # Collect coefficients mapped to feature names
    # Each fold may select different features; collect across all folds
    feat_coefs: dict[str, list[float]] = {}
    for pid in pids:
        fold = folds[pid]
        for fname, coef in zip(fold["selected_features"], fold["group_coef"]):
            feat_coefs.setdefault(fname, []).append(coef)

    # Mean and SD across folds (only features selected in >=50% of folds)
    min_count = len(pids) // 2
    feat_stats: list[tuple[str, float, float, int]] = []
    for fname, coefs in feat_coefs.items():
        if len(coefs) >= min_count:
            feat_stats.append(
                (fname, float(np.mean(coefs)), float(np.std(coefs)), len(coefs)))

    # Sort by absolute mean coefficient
    feat_stats.sort(key=lambda x: abs(x[1]), reverse=True)
    top_k = 20
    feat_stats = feat_stats[:top_k]
    feat_stats.reverse()  # plot bottom-to-top

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(feat_stats))
    names = [f[0] for f in feat_stats]
    means = [f[1] for f in feat_stats]
    sds   = [f[2] for f in feat_stats]
    counts = [f[3] for f in feat_stats]

    colors = ["#d32f2f" if m > 0 else "#1976d2" for m in means]
    bars = ax.barh(y, means, xerr=sds, color=colors, edgecolor="white",
                   linewidth=0.5, capsize=3, alpha=0.85, height=0.7)

    # Annotate with fold count if not all folds
    for i, c in enumerate(counts):
        if c < len(pids):
            ax.text(0.01, y[i], f"({c}/{len(pids)})", va="center",
                    fontsize=6, color="grey", alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean LogReg coefficient (group model)")
    ax.set_title(
        f"Top-{top_k} Feature Weights — Group Model\n"
        f"(mean ± SD across {len(pids)} LOSO folds)",
        fontsize=11, fontweight="bold",
    )

    # Legend for direction
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#d32f2f", label="↑ with high workload"),
        Patch(facecolor="#1976d2", label="↓ with high workload"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Plot 5: Region × category heatmap
# ---------------------------------------------------------------------------

def plot_regional_heatmap(
    coef_data: dict,
    out_path: Path,
) -> None:
    """Pseudo-topographic heatmap of mean coefficient by region and category."""
    folds = coef_data["folds"]
    pids = sorted(folds.keys())

    # Collect signed coefficients per (region, category)
    region_cat_coefs: dict[tuple[str, str], list[float]] = {}
    for pid in pids:
        fold = folds[pid]
        for fname, coef in zip(fold["selected_features"], fold["group_coef"]):
            region, category = _parse_feature(fname)
            region_cat_coefs.setdefault((region, category), []).append(coef)

    # Build matrix
    regions = ["Frontal Mid", "Frontal", "Central", "Parietal", "Occipital", "Connectivity"]
    categories = ["Bandpower", "Ratio", "Hjorth", "Entropy", "Stats", "Aperiodic", "wPLI"]

    # Filter to regions/categories that actually appear
    all_regions = sorted(set(r for (r, _) in region_cat_coefs.keys()))
    all_categories = sorted(set(c for (_, c) in region_cat_coefs.keys()))
    regions    = [r for r in regions if r in all_regions]
    categories = [c for c in categories if c in all_categories]

    matrix = np.full((len(regions), len(categories)), np.nan)
    for ri, reg in enumerate(regions):
        for ci, cat in enumerate(categories):
            vals = region_cat_coefs.get((reg, cat))
            if vals:
                matrix[ri, ci] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions, fontsize=10)

    # Annotate cells
    for ri in range(len(regions)):
        for ci in range(len(categories)):
            val = matrix[ri, ci]
            if not np.isnan(val):
                txt_color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Mean coefficient")
    ax.set_title(
        f"Regional Feature Weight Heatmap — Group Model\n"
        f"(mean across {len(pids)} LOSO folds; red = ↑ with high workload)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Stats: condition-level discrimination
# ---------------------------------------------------------------------------

def print_condition_stats(data: dict[str, list[dict]]) -> None:
    """Paired t-test and Cohen's d for high-cog vs low-cog blocks."""
    pids = sorted(data.keys())
    high_means: list[float] = []
    low_means:  list[float] = []

    for pid in pids:
        pid_high = []
        pid_low  = []
        for trace in _build_block_traces(data[pid]):
            m = float(np.mean(trace["prob_group"]))
            if trace["y_true"] == 1:
                pid_high.append(m)
            else:
                pid_low.append(m)
        if pid_high and pid_low:
            high_means.append(float(np.mean(pid_high)))
            low_means.append(float(np.mean(pid_low)))

    h = np.array(high_means)
    l = np.array(low_means)
    diff = h - l
    d_mean = np.mean(diff)
    d_std  = np.std(diff, ddof=1)
    cohens_d = d_mean / d_std if d_std > 0 else float("nan")

    t_stat, p_val = stats.ttest_rel(h, l)
    w_stat, w_pval = stats.wilcoxon(h, l)

    n_correct = int(np.sum(h > l))

    print("  CONDITION DISCRIMINATION (group model)")
    print("  " + "-" * 50)
    print(f"  n = {len(h)} participants (paired)")
    print(f"  High-cog  mean P(high) = {np.mean(h):.3f} ± {np.std(h):.3f}")
    print(f"  Low-cog   mean P(high) = {np.mean(l):.3f} ± {np.std(l):.3f}")
    print(f"  Difference              = {d_mean:+.3f} ± {d_std:.3f}")
    print(f"  Cohen's d               = {cohens_d:.2f}")
    print(f"  Paired t-test           : t({len(h)-1}) = {t_stat:.3f}, p = {p_val:.4g}")
    print(f"  Wilcoxon signed-rank    : W = {w_stat:.1f}, p = {w_pval:.4g}")
    print(f"  Correct direction       : {n_correct}/{len(h)} "
          f"({100*n_correct/len(h):.0f}%)")
    print()


# ---------------------------------------------------------------------------
# Stats: coefficient consistency across folds
# ---------------------------------------------------------------------------

def print_coefficient_stats(coef_data: dict) -> None:
    """Feature selection frequency and coefficient stability."""
    folds = coef_data["folds"]
    pids = sorted(folds.keys())
    n_folds = len(pids)

    # Count selection frequency and collect coefficients
    feat_coefs: dict[str, list[float]] = {}
    for pid in pids:
        fold = folds[pid]
        for fname, coef in zip(fold["selected_features"], fold["group_coef"]):
            feat_coefs.setdefault(fname, []).append(coef)

    print("  FEATURE COEFFICIENT CONSISTENCY (group model)")
    print("  " + "-" * 75)
    print(f"  {'Feature':<22} {'Sel':>4} {'Mean':>8} {'SD':>7} {'t':>7} {'p':>9} {'Dir':>5}")
    print("  " + "-" * 75)

    rows = []
    for fname, coefs in feat_coefs.items():
        n = len(coefs)
        m = float(np.mean(coefs))
        sd = float(np.std(coefs, ddof=1)) if n > 1 else 0.0
        if sd > 0 and n > 1:
            t_stat, p_val = stats.ttest_1samp(coefs, 0)
        else:
            t_stat, p_val = float("nan"), float("nan")
        direction = "+" if m > 0 else "-"
        rows.append((fname, n, m, sd, t_stat, p_val, direction))

    # Sort by absolute mean coefficient
    rows.sort(key=lambda r: abs(r[2]), reverse=True)

    for fname, n, m, sd, t, p, d in rows:
        sig = "*" if (not np.isnan(p) and p < 0.05) else " "
        print(f"  {fname:<22} {n:>3}/{n_folds} {m:>+8.4f} {sd:>7.4f} "
              f"{t:>7.2f} {p:>8.2e} {d:>4}{sig}")

    # Summary
    always_selected = [f for f, coefs in feat_coefs.items() if len(coefs) == n_folds]
    sig_features = [r[0] for r in rows
                    if not np.isnan(r[5]) and r[5] < 0.05]
    print()
    print(f"  Selected in all {n_folds} folds: {len(always_selected)} features")
    print(f"  Reliably != 0 (p<.05):    {len(sig_features)} features")
    if always_selected:
        print(f"  Always selected: {', '.join(sorted(always_selected))}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beyond-AUC validation plots for LogReg MWL classifier")
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=_DEFAULT_JSON)
    args = parser.parse_args()

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Temporal dynamics (from CSV) ---
    if args.csv.exists():
        print("Loading predictions ...")
        data = load_predictions(args.csv)
        pids = sorted(data.keys())
        n_epochs = sum(len(v) for v in data.values())
        print(f"  {len(pids)} participants, {n_epochs} epochs")
        print()

        plot_group_averaged_traces(
            data,
            _FIG_DIR / "logreg_validation__fig01__temporal_traces_group_avg.png",
        )
        plot_individual_traces(
            data,
            _FIG_DIR / "logreg_validation__fig02__temporal_traces_individual.png",
        )
        plot_block_means(
            data,
            _FIG_DIR / "logreg_validation__fig03__block_mean_predictions.png",
        )
        print()
        print_condition_stats(data)
    else:
        print(f"WARNING: CSV not found at {args.csv}, skipping temporal plots.")

    # --- Neurophysiological plausibility (from JSON) ---
    if args.json.exists():
        print("Loading coefficients ...")
        coef_data = load_coefficients(args.json)
        n_folds = len(coef_data["folds"])
        print(f"  {n_folds} folds")
        print()

        plot_coefficient_bar(
            coef_data,
            _FIG_DIR / "logreg_validation__fig04__coefficient_bar.png",
        )
        plot_regional_heatmap(
            coef_data,
            _FIG_DIR / "logreg_validation__fig05__regional_heatmap.png",
        )
        print()
        print_coefficient_stats(coef_data)
    else:
        print(f"WARNING: JSON not found at {args.json}, skipping coefficient plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
