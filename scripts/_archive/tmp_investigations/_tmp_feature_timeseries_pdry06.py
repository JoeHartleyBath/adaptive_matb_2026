"""Feature time series — FM_Theta, FM_Beta, Par_Alpha for PDRY06.

Computes the three named features using the exact same pipeline as
calibrate_participant.py (EegPreprocessor → 2s windows, 0.25s step →
_extract_feat → z-score with norm_stats) for:

  - rest XDF         (horizontal reference band)
  - calibration C1   (horizontal reference band)
  - calibration C2   (horizontal reference band)
  - control          (time series)
  - adaptation       (time series)

Each feature is plotted in its own panel showing both the raw (un-normed)
value and the z-scored value side-by-side, so we can see whether the 40σ
deviations reflect genuine absolute differences or a tight rest std.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_feature_timeseries_pdry06.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.extract_features import _build_region_map, _extract_feat

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]
FEAT_CFG = _REPO / "config" / "eeg_feature_extraction.yaml"

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
OUT_FIG   = _REPO / "results" / "figures" / "feature_timeseries_pdry06.png"

_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

TARGET_FEATURES = ["FM_Theta", "FM_Beta", "Par_Alpha"]

FILES = {
    "rest":       PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf",
    "cal_c1":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf",
    "cal_c2":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf",
    "control":    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf",
    "adaptation": PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_filter(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XDF, decimate to SRATE, apply EegPreprocessor. Returns (C×N, timestamps)."""
    streams, _ = pyxdf.load_xdf(str(path))
    eeg  = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac  = int(round(actual / SRATE))
            data = data[:, ::fac]
            ts   = ts[::fac]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    data = pp.process(data)
    return data, ts


region_map = _build_region_map(FEAT_CFG, CH_NAMES)

def windowed_features(data: np.ndarray, ts: np.ndarray):
    """Extract features in sliding windows. Returns (feat_matrix N×F, window_end_times)."""
    win  = int(WINDOW_S * SRATE)
    step = int(STEP_S   * SRATE)
    n    = data.shape[1]
    rows = []
    t_ends = []
    for s in range(win, n, step):
        epoch = data[:, s - win : s][np.newaxis, :, :]  # 1 × C × T
        X, names = _extract_feat(epoch, SRATE, region_map)
        rows.append(X[0])
        t_ends.append(ts[min(s, len(ts) - 1)])
    if not rows:
        return np.empty((0, len(names))), np.array([]), names
    return np.vstack(rows), np.array(t_ends), names


# ---------------------------------------------------------------------------
# Compute features for all files
# ---------------------------------------------------------------------------

feat_indices: dict[str, int] = {}
results: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

for cond, path in FILES.items():
    print(f"Processing {cond} ...", flush=True)
    if not path.exists():
        print(f"  NOT FOUND: {path.name} — skipping")
        continue
    data, ts = load_and_filter(path)
    print(f"  {data.shape[1]} samp ({data.shape[1]/SRATE:.0f} s)")
    X, t_ends, names = windowed_features(data, ts)
    results[cond] = (X, t_ends, names)
    if not feat_indices and names:
        for fn in TARGET_FEATURES:
            if fn in names:
                feat_indices[fn] = names.index(fn)
        print(f"  Feature indices: {feat_indices}")

if not feat_indices:
    sys.exit("ERROR: could not locate target features")

# ---------------------------------------------------------------------------
# Summarise rest and calibration as reference distributions
# ---------------------------------------------------------------------------

reference_summary: dict[str, dict[str, tuple[float, float, float]]] = {}
# cond → feat_name → (mean_raw, std_raw, mean_z)

for cond in ("rest", "cal_c1", "cal_c2"):
    if cond not in results:
        continue
    X, _, _ = results[cond]
    reference_summary[cond] = {}
    for fn, fi in feat_indices.items():
        raw_vals = X[:, fi]
        z_vals   = (raw_vals - NORM_MEAN[fi]) / NORM_STD[fi]
        reference_summary[cond][fn] = (
            float(raw_vals.mean()), float(raw_vals.std()),
            float(z_vals.mean()),
        )

print("\nReference distributions (raw feature values):")
print(f"  {'Condition':<10}  {'Feature':<15}  {'mean':>8}  {'std':>8}  {'mean_z':>8}")
for cond, feats in reference_summary.items():
    for fn, (m, s, mz) in feats.items():
        print(f"  {cond:<10}  {fn:<15}  {m:8.4f}  {s:8.4f}  {mz:8.2f}")

print(f"\nnorm_stats std for target features:")
for fn, fi in feat_indices.items():
    print(f"  {fn:<15}  norm_std={NORM_STD[fi]:.6f}  norm_mean={NORM_MEAN[fi]:.6f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

n_feats = len(TARGET_FEATURES)
fig, axes = plt.subplots(n_feats, 2, figsize=(18, 4 * n_feats), constrained_layout=True)
fig.suptitle("PDRY06 — Feature time series: Control vs Adaptation\n"
             "Left = raw (log band-power), Right = z-scored (norm_stats)", fontsize=12)

COND_COLORS = {"control": "#c0392b", "adaptation": "#2980b9"}

for row, fn in enumerate(TARGET_FEATURES):
    fi   = feat_indices[fn]
    ax_r = axes[row, 0]   # raw
    ax_z = axes[row, 1]   # z-scored

    for cond in ("control", "adaptation"):
        if cond not in results:
            continue
        X, t_ends, _ = results[cond]
        t_rel   = t_ends - t_ends[0]
        raw_val = X[:, fi]
        z_val   = (raw_val - NORM_MEAN[fi]) / NORM_STD[fi]

        ax_r.plot(t_rel, raw_val, lw=0.9, color=COND_COLORS[cond],
                  alpha=0.85, label=cond.capitalize())
        ax_z.plot(t_rel, z_val,   lw=0.9, color=COND_COLORS[cond],
                  alpha=0.85, label=cond.capitalize())

    # Reference shading from rest and calibration
    ref_colors = {"rest": "#999999", "cal_c1": "#444444", "cal_c2": "#777777"}
    for ref_cond, ref_color in ref_colors.items():
        if ref_cond not in reference_summary:
            continue
        m, s, mz = reference_summary[ref_cond][fn]
        ax_r.axhspan(m - s, m + s, alpha=0.10, color=ref_color, zorder=0)
        ax_r.axhline(m, color=ref_color, lw=1.0, ls="--", alpha=0.6,
                     label=f"{ref_cond} mean")
        ax_z.axhspan(mz - 1, mz + 1, alpha=0.10, color=ref_color, zorder=0)
        ax_z.axhline(mz, color=ref_color, lw=1.0, ls="--", alpha=0.6,
                     label=f"{ref_cond} mean_z")

    ax_z.axhline(0, color="k", lw=0.6, ls=":", alpha=0.4)

    ax_r.set_title(f"{fn}  [raw]")
    ax_z.set_title(f"{fn}  [z-scored vs rest norm_stats]")
    ax_r.set_ylabel("log band-power")
    ax_z.set_ylabel("z-score (σ)")
    ax_r.set_xlabel("Time (s)")
    ax_z.set_xlabel("Time (s)")
    ax_r.legend(fontsize=7, ncol=3)
    ax_z.legend(fontsize=7, ncol=3)
    ax_r.grid(alpha=0.3)
    ax_z.grid(alpha=0.3)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")
