"""_tmp_feature_dist_pdry06.py

Feature-space forensics for the PDRY06 MWL inference anomaly.

Extracts EEG features (fresh EegPreprocessor per XDF) from all five PDRY06
recordings: rest, cal_c1, cal_c2, control, adaptation.  Features are z-scored
with the PDRY06 norm_stats.json baseline.

Reports:
  - Table of features that deviate >1.5 σ from the calibration mean in
    either control or adaptation, annotated with model-selector status.
  - (a) A heatmap of mean z-scores across conditions × all features,
        with selector-active cells boxed.
  - (b) Boxplots for canonical EEG-MWL features across conditions.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_feature_dist_pdry06.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams  # noqa: E402
from eeg import EegPreprocessor  # noqa: E402
from eeg.online_features import OnlineFeatureExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
OUT_HEAT  = Path(r"C:\adaptive_matb_2026\results\figures\feature_dist_heatmap_pdry06.png")
OUT_BOX   = Path(r"C:\adaptive_matb_2026\results\figures\feature_dist_boxplots_pdry06.png")

meta      = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names: list[str] = meta["channel_names"]
feat_cfg  = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
norm_mean = np.array(ns["mean"], dtype=np.float64)
norm_std  = np.array(ns["std"],  dtype=np.float64)
norm_std[norm_std < 1e-12] = 1.0
threshold = float(json.load(open(MODEL_DIR / "model_config.json"))["youden_threshold"])

selector    = joblib.load(MODEL_DIR / "selector.pkl")
pipeline    = joblib.load(MODEL_DIR / "pipeline.pkl")
selected_idx: set[int] = set(selector.get_support(indices=True).tolist())  # type: ignore[attr-defined]

# Feature names via a dummy call
_ext0 = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
_ext0.compute(np.zeros((len(ch_names), int(WINDOW_S * SRATE))))
feat_names: list[str] = _ext0.feature_names  # type: ignore[assignment]
n_feat = len(feat_names)
print(f"Features: {n_feat}  |  Model-selected: {len(selected_idx)}\n")

# Canonical EEG-MWL features to report in boxplots
FOCUS = [
    "FM_Theta", "FM_Alpha", "FM_Beta",
    "Par_Alpha", "Occ_Alpha", "Cen_Beta",
    "FM_Theta_Alpha", "FM_Theta_Beta", "Cen_Engagement",
    "Cen_HjAct", "Cen_HjComp",
]
FOCUS_IDX = [feat_names.index(fn) for fn in FOCUS if fn in feat_names]

CONDITIONS = [
    ("rest",       "sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf"),
    ("cal_c1",     "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf"),
    ("cal_c2",     "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf"),
    ("control",    "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf"),
    ("adaptation", "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf"),
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(xdf_path: Path) -> np.ndarray:
    """Fresh-filter feature extraction for one XDF. Returns (N, n_feat)."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg = _merge_eeg_streams(streams)
    if eeg is None:
        return np.empty((0, n_feat))
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac  = int(round(actual / SRATE))
            data = data[:, ::fac]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    filtered = pp.process(data)

    ext  = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    step = int(STEP_S   * SRATE)
    win  = int(WINDOW_S * SRATE)
    n    = filtered.shape[1]
    out  = []
    for s in range(win, n - step, step):
        out.append(ext.compute(filtered[:, s - win : s]))
    return np.array(out) if out else np.empty((0, n_feat))


all_raw: dict[str, np.ndarray] = {}
for label, fname in CONDITIONS:
    print(f"  {label} ...", flush=True)
    arr = extract_features(PHYSIO / fname)
    all_raw[label] = arr
    print(f"    {arr.shape[0]} windows")

# ---------------------------------------------------------------------------
# Z-scoring and statistics
# ---------------------------------------------------------------------------

all_z: dict[str, np.ndarray] = {
    k: (v - norm_mean) / norm_std
    for k, v in all_raw.items()
    if v.ndim == 2 and len(v) > 0
}

cond_mean_z: dict[str, np.ndarray] = {k: v.mean(axis=0) for k, v in all_z.items()}

# Training reference from pooled calibration
cal_raw = np.vstack([all_raw["cal_c1"], all_raw["cal_c2"]])
cal_z   = (cal_raw - norm_mean) / norm_std
cal_mean_z = cal_z.mean(axis=0)

# ---------------------------------------------------------------------------
# Deviation table
# ---------------------------------------------------------------------------

DEVIATION_THRESH = 1.5   # |z - cal_mean| flag threshold

ctrl_m_z  = cond_mean_z.get("control",    np.full(n_feat, np.nan))
adapt_m_z = cond_mean_z.get("adaptation", np.full(n_feat, np.nan))
ctrl_dev  = ctrl_m_z  - cal_mean_z
adapt_dev = adapt_m_z - cal_mean_z

flagged = [
    i for i in range(n_feat)
    if abs(ctrl_dev[i]) > DEVIATION_THRESH or abs(adapt_dev[i]) > DEVIATION_THRESH
]

print(f"\nFeatures with |deviation| > {DEVIATION_THRESH} from calibration mean:")
print(f"  (sel=* means feature is selected by PDRY06 model)")
print(f"\n  {'feature':<22} {'sel':>4}  {'cal_z':>7}  {'ctrl_z':>7}  "
      f"{'ctrl_dev':>9}  {'adapt_z':>8}  {'adapt_dev':>9}")
print("  " + "-" * 80)
for i in flagged:
    sel  = "*" if i in selected_idx else " "
    print(f"  {feat_names[i]:<22} {sel:>4}  {cal_mean_z[i]:7.3f}  "
          f"{ctrl_m_z[i]:7.3f}  {ctrl_dev[i]:9.3f}  "
          f"{adapt_m_z[i]:8.3f}  {adapt_dev[i]:9.3f}")

# Also summarise: what's the mean z-score of model-selected features in each condition?
print("\nMean z-score of model-SELECTED features per condition:")
for lbl, mz in cond_mean_z.items():
    sel_mz = mz[sorted(selected_idx)]
    print(f"  {lbl:<12}  mean_z={sel_mz.mean():+.3f}  std_z={sel_mz.std():.3f}")

# ---------------------------------------------------------------------------
# (a) Heatmap: conditions × features, z-score
# ---------------------------------------------------------------------------

cond_order = [c for c, _ in CONDITIONS if c in all_z]
heat = np.array([cond_mean_z[c] for c in cond_order])   # (n_cond, n_feat)

fig1, ax1 = plt.subplots(figsize=(max(14, n_feat * 0.28), 4))
im = ax1.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
ax1.set_xticks(range(n_feat))
ax1.set_xticklabels(feat_names, rotation=90, fontsize=6.5)
ax1.set_yticks(range(len(cond_order)))
ax1.set_yticklabels(cond_order, fontsize=9)

# Box selected features
for i in selected_idx:
    if 0 <= i < n_feat:
        for j in range(len(cond_order)):
            ax1.add_patch(mpatches.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                fill=False, edgecolor="black", lw=0.9, zorder=3,
            ))

plt.colorbar(im, ax=ax1, shrink=0.8, label="z-score (norm_stats baseline)")
ax1.set_title(
    "PDRY06 — Feature z-scores by condition  (black box = model-selected)",
    fontsize=10,
)
plt.tight_layout()
OUT_HEAT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_HEAT, dpi=150, bbox_inches="tight")
print(f"\nHeatmap saved: {OUT_HEAT}")
plt.close(fig1)

# ---------------------------------------------------------------------------
# (b) Boxplots: canonical features, three conditions side-by-side
# ---------------------------------------------------------------------------

n_focus = len(FOCUS_IDX)
ncols   = min(6, n_focus)
nrows   = (n_focus + ncols - 1) // ncols

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 3.2))
axes2_flat  = np.array(axes2).flatten() if nrows * ncols > 1 else [axes2]

box_conds   = ["cal\n(c1+c2)", "control", "adaptation"]
box_sources = [
    np.vstack([all_raw["cal_c1"], all_raw["cal_c2"]]),
    all_raw.get("control",    np.empty((0, n_feat))),
    all_raw.get("adaptation", np.empty((0, n_feat))),
]
box_colors = ["#4c78a8", "#f58518", "#54a24b"]

for plot_i, fi in enumerate(FOCUS_IDX):
    ax = axes2_flat[plot_i]
    fn = feat_names[fi]
    data_list = []
    for src in box_sources:
        if src.ndim == 2 and len(src) > 0:
            data_list.append((src[:, fi] - norm_mean[fi]) / norm_std[fi])
        else:
            data_list.append(np.array([]))

    bp = ax.boxplot(
        [d for d in data_list if len(d) > 0],
        labels=[lbl for lbl, d in zip(box_conds, data_list) if len(d) > 0],
        showfliers=False,
        patch_artist=True,
    )
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    sel_mark = " ★" if fi in selected_idx else ""
    ax.set_title(fn + sel_mark, fontsize=8,
                 fontweight="bold" if fi in selected_idx else "normal")
    ax.axhline(0, color="grey", ls="--", lw=0.6, alpha=0.5)
    ax.set_ylabel("z-score", fontsize=7)
    ax.tick_params(labelsize=7)

for ax in axes2_flat[n_focus:]:
    ax.set_visible(False)

fig2.suptitle(
    "PDRY06 — Canonical feature distributions  (★ = model-selected, z-scored)",
    fontsize=10, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_BOX, dpi=150, bbox_inches="tight")
print(f"Boxplots saved: {OUT_BOX}")
plt.close(fig2)
