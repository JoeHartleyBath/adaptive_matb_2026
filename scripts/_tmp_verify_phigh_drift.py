"""Verify p_high distribution across S005 XDFs using the trained model.

Applies the PSELF model offline to cal_c1, cal_c2, adaptation, and control
XDFs to check whether elevated p_high in adaptation reflects real drift or
model generalisation failure.
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

SRATE = 128.0
WINDOW_S = 2.0
STEP_S = 0.25
model_dir = Path(r"C:\data\adaptive_matb\models\PSELF")
pipeline  = joblib.load(model_dir / "pipeline.pkl")
selector  = joblib.load(model_dir / "selector.pkl")
with open(model_dir / "norm_stats.json") as f:
    ns = json.load(f)
with open(model_dir / "model_config.json") as f:
    model_cfg = json.load(f)
norm_mean  = np.array(ns["mean"])
norm_std   = np.array(ns["std"])
norm_std[norm_std < 1e-12] = 1.0
n_classes  = ns.get("n_classes", 3)
threshold  = model_cfg["youden_threshold"]
print(f"Model: {n_classes}-class  youden_threshold={threshold:.4f}\n")

# ---- Config ----
meta     = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names = meta["channel_names"]
feat_cfg = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

physio = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio")


def p_high_from_xdf(xdf_path: Path):
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        return None, "no EEG streams (merge failed)"

    # _merge_eeg_streams returns time_series as (n_samples, n_ch) — transpose
    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T  # (n_ch, n)
    eeg_ts   = np.array(eeg_stream["time_stamps"])

    # Decimate to 128 Hz if needed (eego streams at native rate)
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > SRATE * 1.1:
            factor = int(round(actual_srate / SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts   = eeg_ts[::factor]

    # Filter — creates a fresh preprocessor per XDF (matches offline training)
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
    ("cal_c1",      "sub-PSELF_ses-S005_task-matb_acq-cal_c1_physio.xdf"),
    ("cal_c2",      "sub-PSELF_ses-S005_task-matb_acq-cal_c2_physio.xdf"),
    ("adaptation",  "sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"),
    ("control",     "sub-PSELF_ses-S005_task-matb_acq-control_physio.xdf"),
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

# Summary comparison
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

# Interpret drift
print()
if results["cal_c1"] is not None and results["adaptation"] is not None:
    gap = results["adaptation"].mean() - results["cal_c1"].mean()
    print(f"  adaptation mean - cal_c1 mean = {gap:+.3f}")
    if abs(gap) < 0.05:
        print("  -> p_high distribution similar between cal_c1 and adaptation.")
        print("     Model generalises well; elevated threshold behaviour is real.")
    elif gap > 0.05:
        print("  -> p_high higher in adaptation than cal_c1.")
        print("     Possible causes: genuine overload, session-level EEG drift,")
        print("     or calibration data not representative of adaptation state.")
    else:
        print("  -> p_high lower in adaptation than cal_c1 (unexpected).")

if results["cal_c1"] is not None and results["cal_c2"] is not None:
    gap_cal = results["cal_c2"].mean() - results["cal_c1"].mean()
    print(f"\n  cal_c2 mean - cal_c1 mean = {gap_cal:+.3f}")
    if abs(gap_cal) > 0.05:
        print("  -> Shift between C1 and C2 confirms within-session EEG drift")
        print("     over the ~26 min gap caused by the failed C2 attempts.")
    else:
        print("  -> C1 and C2 distributions similar; gap not the main driver.")
