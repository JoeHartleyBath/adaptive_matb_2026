"""Verify p_high distribution across PDRY06 XDFs using the trained model.

Applies the PDRY06 model offline to rest, cal_c1, cal_c2, control, and
adaptation XDFs to check whether elevated p_high in adaptation reflects
real EEG drift or a model training failure.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_verify_phigh_drift_pdry06.py
"""
import sys, json, joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import (
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _merge_eeg_streams,
)
from eeg import EegPreprocessor, extract_windows
from eeg.online_features import OnlineFeatureExtractor

import yaml
import pyxdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

model_dir = Path(r"C:\data\adaptive_matb\models\PDRY06")
pipeline  = joblib.load(model_dir / "pipeline.pkl")
selector  = joblib.load(model_dir / "selector.pkl")
with open(model_dir / "norm_stats.json") as f:
    ns = json.load(f)
with open(model_dir / "model_config.json") as f:
    model_cfg = json.load(f)
norm_mean = np.array(ns["mean"])
norm_std  = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0
n_classes = ns.get("n_classes", 3)
threshold = model_cfg["youden_threshold"]
print(f"Model: {n_classes}-class  youden_threshold={threshold:.4f}\n")

# ---- Config ----
meta     = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names = meta["channel_names"]
feat_cfg = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

physio = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")

OUT_FIG = Path(r"C:\adaptive_matb_2026\results\figures\phigh_drift_pdry06.png")


def p_high_from_xdf(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        return None, "no EEG streams (merge failed)"

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T  # (n_ch, n)
    eeg_ts   = np.array(eeg_stream["time_stamps"])

    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > SRATE * 1.1:
            factor = int(round(actual_srate / SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts   = eeg_ts[::factor]

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    filtered = preprocessor.process(eeg_data)

    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)

    step   = int(STEP_S * SRATE)
    win    = int(WINDOW_S * SRATE)
    n_samp = filtered.shape[1]
    p_highs = []
    for start in range(win, n_samp - step, step):
        window    = filtered[:, start - win : start]
        feats     = extractor.compute(window)
        feats_z   = (feats - norm_mean) / norm_std
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        proba     = pipeline.predict_proba(feats_sel)[0]
        p_highs.append(float(proba[-1]))

    return np.array(p_highs), None


files = [
    ("rest",        "sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf"),
    ("cal_c1",      "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf"),
    ("cal_c2",      "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf"),
    ("control",     "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf"),
    ("adaptation",  "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf"),
]

results = {}
for label, fname in files:
    print(f"Processing {label} ...", flush=True)
    ph, err = p_high_from_xdf(physio / fname)
    if err:
        print(f"  ERROR: {err}")
        results[label] = None
    else:
        above = (ph > threshold).mean() * 100
        results[label] = ph
        print(
            f"  n={len(ph):4d}  mean={ph.mean():.3f}  std={ph.std():.3f}  "
            f"median={np.median(ph):.3f}  "
            f"pct_above_thr({threshold:.3f})={above:.1f}%"
        )

# ---- Summary table ----
print("\n" + "=" * 65)
print(f"  {'Recording':<14s}  {'mean':>6s}  {'std':>6s}  {'median':>7s}  {'%>thr':>6s}")
print(f"  {'-'*14}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}")
for label, ph in results.items():
    if ph is None:
        print(f"  {label:<14s}  ERROR")
    else:
        above = (ph > threshold).mean() * 100
        print(
            f"  {label:<14s}  {ph.mean():6.3f}  {ph.std():6.3f}  "
            f"{np.median(ph):7.3f}  {above:5.1f}%"
        )

# ---- Interpretation ----
print()
cal_c1_ph = results.get("cal_c1")
adapt_ph  = results.get("adaptation")

if cal_c1_ph is not None and adapt_ph is not None:
    gap = adapt_ph.mean() - cal_c1_ph.mean()
    print(f"  adaptation mean - cal_c1 mean = {gap:+.3f}")
    if abs(gap) < 0.05:
        print("  -> Similar distribution: elevated p_high present already in cal_c1.")
        print("     Likely model training failure (near-ceiling predictions).")
    elif gap > 0.05:
        print("  -> p_high higher in adaptation than cal_c1.")
        print("     Possible causes: genuine overload, EEG drift, or model not")
        print("     representative of adaptation state.")
    else:
        print("  -> p_high lower in adaptation than cal_c1 (unexpected).")

cal_c2_ph = results.get("cal_c2")
if cal_c1_ph is not None and cal_c2_ph is not None:
    gap_cal = cal_c2_ph.mean() - cal_c1_ph.mean()
    print(f"\n  cal_c2 mean - cal_c1 mean = {gap_cal:+.3f}")
    if abs(gap_cal) > 0.05:
        print("  -> Shift between C1 and C2 suggests within-session EEG drift.")
    else:
        print("  -> C1 and C2 distributions similar; no significant drift between calibrations.")

# ---- Plot ----
labels_ordered = [lbl for lbl, _ in files]
data_to_plot   = [results[lbl] for lbl in labels_ordered]
colors = ["#4e8bc4", "#e8913a", "#e8913a", "#6bb86b", "#d05c5c"]

fig, ax = plt.subplots(figsize=(9, 5))

valid_idx   = [i for i, d in enumerate(data_to_plot) if d is not None]
valid_data  = [data_to_plot[i] for i in valid_idx]
valid_labels = [labels_ordered[i] for i in valid_idx]
valid_colors = [colors[i] for i in valid_idx]

parts = ax.violinplot(valid_data, positions=range(len(valid_data)), showmedians=True, widths=0.7)
for pc, col in zip(parts["bodies"], valid_colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.6)
for part_name in ("cbars", "cmins", "cmaxes", "cmedians"):
    if part_name in parts:
        parts[part_name].set_color("black")
        parts[part_name].set_linewidth(1.2)

ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
           label=f"Youden threshold ({threshold:.3f})")
ax.set_xticks(range(len(valid_labels)))
ax.set_xticklabels(valid_labels, fontsize=11)
ax.set_ylabel("p(HIGH)", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_title("PDRY06 — p_high distribution per recording\n(model applied offline)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")
