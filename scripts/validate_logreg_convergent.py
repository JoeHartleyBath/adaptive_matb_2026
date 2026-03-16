"""Convergent validity: classifier predictions vs subjective/behavioral measures.

Recovers per-block VR-TSST condition labels from HDF5 metadata, computes
per-(participant, condition) mean P(high), and merges with subjective ratings
and behavioral performance from the VR-TSST aggregated CSVs.

Produces:
  Fig 06 - 4-condition P(high) boxplot (Stress x CogLoad)
  Fig 07 - Repeated-measures correlation: P(high) vs NASA Mental Demand
  Fig 08 - Repeated-measures correlation: P(high) vs Response Accuracy

Usage:
    python -m scripts.validate_logreg_convergent
    python -m scripts.validate_logreg_convergent --h5-dir <path> --subj <path> --behav <path>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_FIG_DIR = _ROOT / "results" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_CSV = _ROOT / "results" / "test_pretrain" / "logreg_epoch_predictions.csv"
_DEFAULT_H5_DIR = Path(r"c:\vr_tsst_2025\output\matb_pretrain\continuous")
_DEFAULT_SUBJ = Path(r"c:\vr_tsst_2025\output\aggregated\subjective.csv")
_DEFAULT_BEHAV = Path(r"c:\vr_tsst_2025\output\aggregated\all_data_aggregated.csv")

# Normalise HDF5 condition names (strip embedded digits, e.g. HighCog1022)
_DIGIT_TAG = re.compile(r"\d+")


def _normalise_condition(raw: str) -> str:
    """'HighStress_HighCog1022_Task' -> 'HighStress_HighCog_Task'."""
    parts = raw.split("_")
    return "_".join(_DIGIT_TAG.sub("", p) for p in parts)


# ---------------------------------------------------------------------------
# 1. Build per-(pid, condition) mean P(high)
# ---------------------------------------------------------------------------

def build_condition_predictions(
    csv_path: Path,
    h5_dir: Path,
) -> pd.DataFrame:
    """Return DataFrame with columns: pid, pid_int, condition, stress, cogload,
    mean_prob_group, mean_prob_personal."""

    pred = pd.read_csv(csv_path)
    pids = sorted(pred["pid"].unique())

    rows: list[dict] = []
    for pid in pids:
        h5_path = h5_dir / f"{pid}.h5"
        if not h5_path.exists():
            print(f"  WARNING: {h5_path} not found, skipping {pid}")
            continue

        with h5py.File(h5_path, "r") as h5:
            raw_conditions = json.loads(h5.attrs["task_conditions"])

        # block_idx -> condition name (normalised)
        block_to_cond = {
            i: _normalise_condition(c)
            for i, c in enumerate(raw_conditions)
        }

        pid_df = pred[pred["pid"] == pid]
        for bidx, cond in block_to_cond.items():
            block = pid_df[pid_df["block_idx"] == bidx]
            if block.empty:
                continue
            rows.append({
                "pid": pid,
                "pid_int": int(pid[1:]),
                "condition": cond,
                "stress": "High" if "HighStress" in cond else "Low",
                "cogload": "High" if "HighCog" in cond else "Low",
                "mean_prob_group": float(block["y_prob_group"].mean()),
                "mean_prob_personal": float(block["y_prob_personal"].mean()),
                "n_epochs": len(block),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Merge with subjective + behavioral
# ---------------------------------------------------------------------------

def merge_external(
    cond_df: pd.DataFrame,
    subj_path: Path,
    behav_path: Path,
) -> pd.DataFrame:
    """Merge condition-level predictions with subjective and behavioral data."""

    subj = pd.read_csv(subj_path)
    subj_cols = [
        "Participant_ID", "Condition",
        "NASA_Mental", "NASA_Performance", "NASA_Effort",
        "Stress", "Arousal",
    ]
    subj = subj[[c for c in subj_cols if c in subj.columns]]

    merged = cond_df.merge(
        subj,
        left_on=["pid_int", "condition"],
        right_on=["Participant_ID", "Condition"],
        how="left",
    )

    # Behavioral data – filter to task conditions only
    behav = pd.read_csv(behav_path)
    task_conds = [c for c in behav["Condition"].unique() if c.endswith("_Task")]
    behav = behav[behav["Condition"].isin(task_conds)]
    behav_cols = [
        "Participant_ID", "Condition",
        "Full_Response_Accuracy", "Full_Mean_Response_Latency_sec",
        "Full_Response_Rate_per_min",
    ]
    behav = behav[[c for c in behav_cols if c in behav.columns]]

    merged = merged.merge(
        behav,
        left_on=["pid_int", "condition"],
        right_on=["Participant_ID", "Condition"],
        how="left",
        suffixes=("", "_behav"),
    )

    # Clean up duplicate join columns
    for col in ["Participant_ID", "Condition", "Participant_ID_behav", "Condition_behav"]:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)

    return merged


# ---------------------------------------------------------------------------
# 3. Repeated-measures correlation (manual implementation)
# ---------------------------------------------------------------------------

def rmcorr(
    df: pd.DataFrame,
    x: str,
    y: str,
    subject: str = "pid",
) -> dict:
    """Compute repeated-measures correlation (Bakdash & Marusich, 2017).

    Returns dict with r, ci_lo, ci_hi, t, p, df_error, n_subjects, n_obs.
    """
    sub = df[[subject, x, y]].dropna()
    subjects = sub[subject].unique()
    n_sub = len(subjects)
    n_obs = len(sub)

    if n_sub < 3 or n_obs < 6:
        return {"r": float("nan"), "p": float("nan"), "n_subjects": n_sub, "n_obs": n_obs}

    # Remove per-subject means (partial out subject)
    x_vals = sub[x].values.astype(float)
    y_vals = sub[y].values.astype(float)
    sub_ids = sub[subject].values

    x_resid = np.empty_like(x_vals)
    y_resid = np.empty_like(y_vals)
    for s in subjects:
        mask = sub_ids == s
        x_resid[mask] = x_vals[mask] - x_vals[mask].mean()
        y_resid[mask] = y_vals[mask] - y_vals[mask].mean()

    # Guard: if either residual is constant, correlation is undefined
    if np.std(x_resid) == 0 or np.std(y_resid) == 0:
        return {"r": float("nan"), "p": float("nan"), "n_subjects": n_sub, "n_obs": n_obs}

    # Pearson on residuals
    r, _ = sp_stats.pearsonr(x_resid, y_resid)

    # Degrees of freedom: n_obs - n_sub - 1
    df_err = n_obs - n_sub - 1
    t_stat = r * np.sqrt(df_err / (1 - r**2)) if abs(r) < 1 else float("inf")
    p_val = 2 * sp_stats.t.sf(abs(t_stat), df_err) if df_err > 0 else float("nan")

    # Fisher z 95% CI
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(df_err - 1) if df_err > 1 else float("nan")
    ci_lo = float(np.tanh(z - 1.96 * se))
    ci_hi = float(np.tanh(z + 1.96 * se))

    return {
        "r": float(r),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "t": float(t_stat),
        "p": float(p_val),
        "df_error": int(df_err),
        "n_subjects": n_sub,
        "n_obs": n_obs,
    }


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------

def plot_4condition_boxplot(df: pd.DataFrame, out: Path) -> None:
    """Fig 06: P(high) across 4 VR-TSST conditions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    conditions = [
        ("Low", "Low"),
        ("Low", "High"),
        ("High", "Low"),
        ("High", "High"),
    ]
    labels = [
        "LowStress\nLowCog",
        "LowStress\nHighCog",
        "HighStress\nLowCog",
        "HighStress\nHighCog",
    ]
    colours = ["#4daf4a", "#e41a1c", "#4daf4a", "#e41a1c"]

    box_data = []
    for stress, cog in conditions:
        mask = (df["stress"] == stress) & (df["cogload"] == cog)
        box_data.append(df.loc[mask, "mean_prob_group"].values)

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)

    # Overlay individual dots
    rng = np.random.default_rng(42)
    for i, vals in enumerate(box_data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(
            np.full_like(vals, i + 1) + jitter,
            vals,
            alpha=0.5, s=18, c="k", zorder=3,
        )

    ax.set_ylabel("Mean P(high MWL)")
    ax.set_title("Classifier predictions across VR-TSST conditions")
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(_ROOT)}")


_COND_COLOURS = {
    ("Low", "Low"):   "#4daf4a",   # green
    ("Low", "High"):  "#e41a1c",   # red
    ("High", "Low"):  "#377eb8",   # blue
    ("High", "High"): "#984ea3",   # purple
}
_COND_LABELS = {
    ("Low", "Low"):   "LowS / LowC",
    ("Low", "High"):  "LowS / HighC",
    ("High", "Low"):  "HighS / LowC",
    ("High", "High"): "HighS / HighC",
}


def plot_rmcorr_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
) -> dict:
    """Scatter colour-coded by condition with common-slope rmcorr line."""
    rc = rmcorr(df, x, y, subject="pid")

    fig, ax = plt.subplots(figsize=(6.5, 5))

    valid = df.dropna(subset=[x, y])

    # -- Colour-coded scatter by condition --
    for (stress, cog), colour in _COND_COLOURS.items():
        mask = (valid["stress"] == stress) & (valid["cogload"] == cog)
        sub = valid[mask]
        if sub.empty:
            continue
        ax.scatter(
            sub[x], sub[y],
            c=colour, alpha=0.6, s=28, zorder=3, edgecolors="white",
            linewidths=0.3, label=_COND_LABELS[(stress, cog)],
        )

    # -- Common-slope regression line (rmcorr fit) --
    # Compute within-subject-centred slope, then overlay using grand means
    x_vals = valid[x].values.astype(float)
    y_vals = valid[y].values.astype(float)
    pids = valid["pid"].values

    x_resid = np.empty_like(x_vals)
    y_resid = np.empty_like(y_vals)
    for pid in np.unique(pids):
        m = pids == pid
        x_resid[m] = x_vals[m] - x_vals[m].mean()
        y_resid[m] = y_vals[m] - y_vals[m].mean()

    if np.std(x_resid) > 0:
        slope = np.dot(x_resid, y_resid) / np.dot(x_resid, x_resid)
        intercept = np.mean(y_vals) - slope * np.mean(x_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                c="k", lw=2, ls="--", alpha=0.7, zorder=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.8, loc="best")

    # Annotation
    sig = "*" if rc["p"] < 0.05 else ""
    ax.text(
        0.02, 0.98,
        f"rmcorr = {rc['r']:.3f} [{rc['ci_lo']:.3f}, {rc['ci_hi']:.3f}]\n"
        f"p = {rc['p']:.4g}{sig}  (n={rc['n_subjects']}, obs={rc['n_obs']})",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(_ROOT)}")
    return rc


# ---------------------------------------------------------------------------
# 5. Stats summary
# ---------------------------------------------------------------------------

def print_convergent_stats(df: pd.DataFrame) -> None:
    """Print convergent validity statistics to console."""
    print()
    print("  CONVERGENT VALIDITY STATISTICS")
    print("  " + "=" * 60)

    # -- 4-condition descriptives --
    print()
    print("  Per-condition mean P(high) [group model]:")
    print("  " + "-" * 50)
    for stress in ["Low", "High"]:
        for cog in ["Low", "High"]:
            mask = (df["stress"] == stress) & (df["cogload"] == cog)
            vals = df.loc[mask, "mean_prob_group"]
            print(f"  {stress}Stress_{cog}Cog: "
                  f"{vals.mean():.3f} ± {vals.std():.3f}  (n={len(vals)})")

    # -- CogLoad main effect (collapsed across stress) --
    print()
    high_cog = df[df["cogload"] == "High"].groupby("pid")["mean_prob_group"].mean()
    low_cog = df[df["cogload"] == "Low"].groupby("pid")["mean_prob_group"].mean()
    common = sorted(set(high_cog.index) & set(low_cog.index))
    h = high_cog.loc[common].values
    l = low_cog.loc[common].values
    t_cog, p_cog = sp_stats.ttest_rel(h, l)
    d_cog = (h - l).mean() / (h - l).std(ddof=1)
    print(f"  CogLoad main effect (High vs Low, collapsed across Stress):")
    print(f"    High = {h.mean():.3f} ± {h.std():.3f}, Low = {l.mean():.3f} ± {l.std():.3f}")
    print(f"    t({len(common)-1}) = {t_cog:.3f}, p = {p_cog:.4g}, d = {d_cog:.2f}")

    # -- Stress main effect --
    high_stress = df[df["stress"] == "High"].groupby("pid")["mean_prob_group"].mean()
    low_stress = df[df["stress"] == "Low"].groupby("pid")["mean_prob_group"].mean()
    common_s = sorted(set(high_stress.index) & set(low_stress.index))
    hs = high_stress.loc[common_s].values
    ls = low_stress.loc[common_s].values
    t_s, p_s = sp_stats.ttest_rel(hs, ls)
    d_s = (hs - ls).mean() / (hs - ls).std(ddof=1) if (hs - ls).std(ddof=1) > 0 else float("nan")
    print()
    print(f"  Stress main effect (High vs Low, collapsed across CogLoad):")
    print(f"    High = {hs.mean():.3f} ± {hs.std():.3f}, Low = {ls.mean():.3f} ± {ls.std():.3f}")
    print(f"    t({len(common_s)-1}) = {t_s:.3f}, p = {p_s:.4g}, d = {d_s:.2f}")

    # -- rmcorr summaries --
    print()
    print("  Repeated-measures correlations with P(high):")
    print("  " + "-" * 60)
    measures = [
        ("NASA_Mental", "NASA Mental Demand"),
        ("NASA_Effort", "NASA Effort"),
        ("NASA_Performance", "NASA Performance"),
        ("Arousal", "Self-reported Arousal"),
        ("Full_Response_Accuracy", "Response Accuracy"),
        ("Full_Mean_Response_Latency_sec", "Response Latency (s)"),
        ("Full_Response_Rate_per_min", "Response Rate (/min)"),
    ]
    for col, label in measures:
        if col not in df.columns:
            continue
        valid = df[["pid", "mean_prob_group", col]].dropna()
        if len(valid) < 6:
            continue
        rc = rmcorr(valid, "mean_prob_group", col, subject="pid")
        if np.isnan(rc["r"]):
            print(f"  {label:<30s} — insufficient variance")
            continue
        sig = "*" if rc["p"] < 0.05 else " "
        print(f"  {label:<30s} r={rc['r']:+.3f} "
              f"[{rc['ci_lo']:.3f},{rc['ci_hi']:.3f}] "
              f"p={rc['p']:.4g}{sig}  n={rc['n_subjects']} obs={rc['n_obs']}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--h5-dir", type=Path, default=_DEFAULT_H5_DIR)
    parser.add_argument("--subj", type=Path, default=_DEFAULT_SUBJ)
    parser.add_argument("--behav", type=Path, default=_DEFAULT_BEHAV)
    args = parser.parse_args()

    # -- Build condition-level predictions --
    print("Building condition-level predictions ...")
    cond_df = build_condition_predictions(args.csv, args.h5_dir)
    print(f"  {len(cond_df)} rows ({cond_df['pid'].nunique()} participants × "
          f"{cond_df['condition'].nunique()} conditions)")

    # -- Merge with external measures --
    print("Merging with subjective + behavioral data ...")
    merged = merge_external(cond_df, args.subj, args.behav)
    n_subj = merged["NASA_Mental"].notna().sum()
    n_behav = merged["Full_Response_Accuracy"].notna().sum()
    print(f"  Subjective match: {n_subj}/{len(merged)} rows")
    print(f"  Behavioral match: {n_behav}/{len(merged)} rows")

    # -- Fig 06: 4-condition boxplot --
    plot_4condition_boxplot(
        merged,
        _FIG_DIR / "logreg_validation__fig06__4condition_boxplot.png",
    )

    # -- Fig 07: rmcorr P(high) vs NASA Mental Demand --
    if "NASA_Mental" in merged.columns and merged["NASA_Mental"].notna().any():
        rc_mental = plot_rmcorr_scatter(
            merged, "mean_prob_group", "NASA_Mental",
            xlabel="Mean P(high MWL)",
            ylabel="NASA Mental Demand",
            title="Classifier vs subjective mental workload",
            out=_FIG_DIR / "logreg_validation__fig07__rmcorr_nasa_mental.png",
        )

    # -- Fig 08: rmcorr P(high) vs Response Accuracy --
    if "Full_Response_Accuracy" in merged.columns and merged["Full_Response_Accuracy"].notna().any():
        rc_acc = plot_rmcorr_scatter(
            merged, "mean_prob_group", "Full_Response_Accuracy",
            xlabel="Mean P(high MWL)",
            ylabel="Response Accuracy",
            title="Classifier vs behavioral accuracy",
            out=_FIG_DIR / "logreg_validation__fig08__rmcorr_accuracy.png",
        )

    # -- Print all stats --
    print_convergent_stats(merged)

    print("Done.")


if __name__ == "__main__":
    main()
