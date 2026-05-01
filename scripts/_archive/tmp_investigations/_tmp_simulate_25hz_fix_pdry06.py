"""Simulation: does removing the 25 Hz artifact fix the PDRY06 p_high saturation?

Three variants are compared for the control and adaptation sessions:

  A. Deployed model    + deployed LORO threshold (0.4849)   [what actually ran]
  B. Deployed model    + 10-fold CV threshold    (0.3285)   [threshold-only fix]
  C. 25 Hz-notched model + 10-fold CV threshold             [full fix simulation]

For variant C the full calibration pipeline is re-run on notched EEG:
  - narrow IIR notch at 25 Hz (Q=50) applied AFTER the standard bandpass/50 Hz notch
  - rest XDF is also notched before recomputing norm_stats (mean + std shift too)
  - SelectKBest(k=35) + StandardScaler + SVC(linear, C=1) re-fitted from notched cal XDFs
  - 10-fold stratified CV threshold re-derived from notched calibration features

This is a DIAGNOSTIC SIMULATION ONLY.  The 25 Hz notch is NOT added to the
production pipeline (it bisects the beta band).  Goal: confirm that the spike
is the root cause before committing to a physical lab fix.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_simulate_25hz_fix_pdry06.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import scipy.signal
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _merge_eeg_streams,
    _extract_all_blocks,
    _find_block_bounds,
    _find_stream,
    _load_eeg_metadata,
    _parse_markers,
)
from eeg import EegPreprocessor, extract_windows, slice_block  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402
from ml.dataset import LABEL_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE     = 128.0
WIN_S     = WINDOW_CONFIG.window_s
STEP_S    = WINDOW_CONFIG.step_s
CAL_K     = 35
CAL_C     = 1.0
SEED      = 42
CV_SPLITS = 10
NOTCH_Q   = 50.0   # narrow notch at 25 Hz — diagnostic only

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"
OUT_FIG   = _REPO / "results" / "figures" / "sim_25hz_fix_pdry06.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

FILES = {
    "rest":       PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf",
    "cal_c1":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf",
    "cal_c2":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf",
    "control":    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf",
    "adaptation": PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf",
}

DEPLOYED_LORO_THR    = 0.484876   # from model_config.json (what ran live)
KFOLD_THR_UNNOTCHED  = 0.3285     # from _tmp_threshold_diagnostic_pdry06.py

# ---------------------------------------------------------------------------
# Step 0 — Load deployed artefacts (Variant A & B)
# ---------------------------------------------------------------------------
print("=" * 72)
print("PDRY06 25 Hz ARTIFACT SIMULATION")
print("=" * 72)

_ns = json.loads((MODEL_DIR / "norm_stats.json").read_text())
DEPLOYED_MEAN = np.array(_ns["mean"])
DEPLOYED_STD  = np.array(_ns["std"])
DEPLOYED_STD[DEPLOYED_STD < 1e-12] = 1.0

_pipe    = joblib.load(MODEL_DIR / "pipeline.pkl")
_sel     = joblib.load(MODEL_DIR / "selector.pkl")
print(f"  Loaded pipeline.pkl and selector.pkl from {MODEL_DIR}")

region_map = _build_region_map(FEAT_CFG, CH_NAMES)

# ---------------------------------------------------------------------------
# Helper: apply narrow IIR notch at 25 Hz to a C×N signal
# ---------------------------------------------------------------------------
def apply_25hz_notch(data: np.ndarray, fs: float = SRATE, Q: float = NOTCH_Q) -> np.ndarray:
    """Zero-phase IIR notch at 25 Hz. Input/output: C×N float32 array."""
    b, a = scipy.signal.iirnotch(25.0, Q=Q, fs=fs)
    sos  = scipy.signal.tf2sos(b, a)
    out  = scipy.signal.sosfiltfilt(sos, data, axis=1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: load XDF, decimate, apply EegPreprocessor (optionally + 25 Hz notch)
# ---------------------------------------------------------------------------
def load_eeg(path: Path, notch25: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Returns (C×N filtered data, timestamps)."""
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
    if notch25:
        data = apply_25hz_notch(data)
    return data, ts


# ---------------------------------------------------------------------------
# Helper: windowed p_high inference using a fitted selector + pipeline
# ---------------------------------------------------------------------------
def windowed_phigh(
    data:      np.ndarray,
    ts:        np.ndarray,
    sel:       SelectKBest,
    pipeline,            # sklearn Pipeline (StandardScaler + SVC)
    norm_mean: np.ndarray,
    norm_std:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a 2-s window with 0.25-s step; return (p_high array, window_end_times)."""
    win  = int(WIN_S  * SRATE)
    step = int(STEP_S * SRATE)
    n    = data.shape[1]
    p_list: list[float] = []
    t_list: list[float] = []
    for s in range(win, n, step):
        epoch = data[:, s - win : s][np.newaxis, :, :]   # 1 × C × T
        X, _  = _extract_feat(epoch, SRATE, region_map)
        X_norm = (X - norm_mean) / norm_std
        X_sel  = sel.transform(X_norm)
        p_high = pipeline.predict_proba(X_sel)[0, -1]
        p_list.append(float(p_high))
        t_list.append(float(ts[min(s, len(ts) - 1)]))
    return np.array(p_list), np.array(t_list)


# ---------------------------------------------------------------------------
# Step 1 — Variants A & B: windowed inference with deployed artefacts
# ---------------------------------------------------------------------------
print("\n--- Step 1: Variants A & B (deployed model, standard preprocessing) ---")

var_p: dict[str, dict[str, np.ndarray]] = {}
var_t: dict[str, dict[str, np.ndarray]] = {}

for cond in ("control", "adaptation"):
    print(f"  Loading {cond} ...", flush=True)
    data, ts = load_eeg(FILES[cond], notch25=False)
    p, t = windowed_phigh(data, ts, _sel, _pipe, DEPLOYED_MEAN, DEPLOYED_STD)
    var_p.setdefault("A", {})[cond] = p
    var_t.setdefault("A", {})[cond] = t
    var_p.setdefault("B", {})[cond] = p   # same inference, different threshold at plot time
    var_t.setdefault("B", {})[cond] = t
    print(f"    n_windows={len(p)}  mean={p.mean():.3f}  std={p.std():.3f}  "
          f"% > LORO thr ({DEPLOYED_LORO_THR:.3f}): {(p > DEPLOYED_LORO_THR).mean():.1%}  "
          f"% > CV thr ({KFOLD_THR_UNNOTCHED:.3f}): {(p > KFOLD_THR_UNNOTCHED).mean():.1%}")

# ---------------------------------------------------------------------------
# Step 2 — Variant C: load REST with 25 Hz notch, recompute norm_stats
# ---------------------------------------------------------------------------
print("\n--- Step 2: Load REST (notched) — recompute norm_stats ---")

REST_SETTLE_S = 5.0   # match _load_rest_xdf_block

rest_data, rest_ts = load_eeg(FILES["rest"], notch25=True)
print(f"  rest data: {rest_data.shape}  ({rest_data.shape[1]/SRATE:.0f} s)")

# Use full XDF (no REST markers saved) minus settle — same as _load_rest_xdf_block fallback
settle_idx = int(REST_SETTLE_S * SRATE)
rest_data  = rest_data[:, settle_idx:]
rest_ts    = rest_ts[settle_idx:]

# Extract windows (same parameters as calibration pipeline)
win_samp  = int(WIN_S  * SRATE)
step_samp = int(STEP_S * SRATE)
n_rest    = rest_data.shape[1]
rest_epochs: list[np.ndarray] = []
for s in range(win_samp, n_rest, step_samp):
    rest_epochs.append(rest_data[:, s - win_samp : s][np.newaxis, :, :])
if not rest_epochs:
    sys.exit("ERROR: no rest windows available")
rest_epoch_arr = np.concatenate(rest_epochs, axis=0)
print(f"  rest windows: {rest_epoch_arr.shape[0]}")

X_rest, feat_names = _extract_feat(rest_epoch_arr, SRATE, region_map)
NOTCH_MEAN = X_rest.mean(axis=0)
NOTCH_STD  = X_rest.std(axis=0)
NOTCH_STD[NOTCH_STD < 1e-12] = 1.0

print(f"  norm_stats recomputed: n_features={len(NOTCH_MEAN)}")
# Show FM_Beta before/after notch
fm_beta_idx = feat_names.index("FM_Beta") if "FM_Beta" in feat_names else None
if fm_beta_idx is not None:
    print(f"  FM_Beta: deployed mean={DEPLOYED_MEAN[fm_beta_idx]:.5f}  std={DEPLOYED_STD[fm_beta_idx]:.5f}")
    print(f"  FM_Beta: notched  mean={NOTCH_MEAN[fm_beta_idx]:.5f}  std={NOTCH_STD[fm_beta_idx]:.5f}")

# ---------------------------------------------------------------------------
# Step 3 — Variant C: load calibration XDFs with 25 Hz notch, re-fit model
# ---------------------------------------------------------------------------
print("\n--- Step 3: Load calibration XDFs (notched) and re-fit model ---")

import os
import re
_SCENARIOS_DIR = _REPO / "experiment" / "scenarios"
_MATB_SCENARIO_OFFSET_S = 12.0
import calibrate_participant as _cal_mod  # noqa: E402 — for _parse_scenario_blocks etc.

CAL_XDFS = [FILES["cal_c1"], FILES["cal_c2"]]

xdf_X_notch: list[np.ndarray] = []
xdf_y_notch: list[np.ndarray] = []

for xdf_path in CAL_XDFS:
    acq_tag = "C1" if "c1" in xdf_path.stem else "C2"
    print(f"  [{acq_tag}] {xdf_path.name}", flush=True)

    data, ts = load_eeg(xdf_path, notch25=True)

    # Reconstruct marker-based block timing from scenario file
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    marker_stream = _find_stream(streams, "Markers")
    markers = _parse_markers(marker_stream)
    block_specs: list[tuple[float, float, str]] = _extract_all_blocks(markers)

    if not block_specs:
        scenario_path = _cal_mod._find_calibration_scenario(xdf_path, _SCENARIOS_DIR)
        if scenario_path is None:
            print(f"    SKIPPED (no markers, no scenario)")
            continue
        scenario_blocks = _cal_mod._parse_scenario_blocks(scenario_path)
        offset_s = float(os.environ.get("MATB_SCENARIO_OFFSET_S", str(_MATB_SCENARIO_OFFSET_S)))
        matb_t0 = ts[0] + offset_s
        block_specs = [(matb_t0 + s, matb_t0 + e, lvl) for s, e, lvl in scenario_blocks]
        print(f"    scenario fallback: {len(block_specs)} blocks")

    all_feats: list[np.ndarray] = []
    all_labels: list[int] = []
    for start_t, end_t, level in block_specs:
        s_idx = int(np.searchsorted(ts, start_t))
        e_idx = int(np.searchsorted(ts, end_t))
        block = slice_block(data, s_idx, e_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] == 0:
            continue
        X_block, _ = _extract_feat(epochs, SRATE, region_map)
        X_norm = (X_block - NOTCH_MEAN) / NOTCH_STD
        lbl = LABEL_MAP.get(level)
        if lbl is None:
            continue
        all_feats.append(X_norm)
        all_labels.extend([lbl] * epochs.shape[0])
        print(f"    level={level:<10}  n_windows={epochs.shape[0]}")

    if not all_feats:
        print(f"    SKIPPED (no usable blocks)")
        continue
    xdf_X_notch.append(np.concatenate(all_feats))
    xdf_y_notch.append(np.array(all_labels, dtype=np.int64))

if len(xdf_X_notch) < 2:
    print("ERROR: fewer than 2 cal XDFs loaded — cannot re-fit model")
    sys.exit(1)

cal_X_notch = np.concatenate(xdf_X_notch)
cal_y_notch = np.concatenate(xdf_y_notch)
print(f"\n  Combined notched cal: n={len(cal_y_notch)}  "
      f"LOW={(cal_y_notch==LABEL_MAP['LOW']).sum()}  "
      f"MOD={(cal_y_notch==LABEL_MAP.get('MODERATE',1)).sum()}  "
      f"HIGH={(cal_y_notch==LABEL_MAP['HIGH']).sum()}")

# Fit final model (same parameters as calibrate_participant.py)
k = min(CAL_K, cal_X_notch.shape[1])
NOTCH_SEL = SelectKBest(f_classif, k=k)
cal_X_sel = NOTCH_SEL.fit_transform(cal_X_notch, cal_y_notch)
NOTCH_SC  = StandardScaler()
cal_X_sc  = NOTCH_SC.fit_transform(cal_X_sel)
NOTCH_SVC = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
NOTCH_SVC.fit(cal_X_sc, cal_y_notch)
print("  Notched SVC fitted.")

# Compute 10-fold CV threshold from notched cal features
print("\n--- Step 3b: 10-fold CV threshold (notched calibration) ---")
high_label = LABEL_MAP["HIGH"]
y_bin = (cal_y_notch == high_label).astype(int)
cv    = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
pool_p_notch = np.zeros(len(cal_y_notch))
fold_Js: list[float] = []

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(cal_X_notch, cal_y_notch)):
    Xtr, ytr = cal_X_notch[tr_idx], cal_y_notch[tr_idx]
    Xte, yte = cal_X_notch[te_idx], cal_y_notch[te_idx]
    k_f = min(CAL_K, Xtr.shape[1])
    sel_f = SelectKBest(f_classif, k=k_f)
    Xtr_s = sel_f.fit_transform(Xtr, ytr)
    sc_f  = StandardScaler()
    Xtr_s = sc_f.fit_transform(Xtr_s)
    svc_f = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
    svc_f.fit(Xtr_s, ytr)
    Xte_s = sc_f.transform(sel_f.transform(Xte))
    p_f   = svc_f.predict_proba(Xte_s)[:, -1]
    pool_p_notch[te_idx] = p_f
    y_bin_f = (yte == high_label).astype(int)
    if y_bin_f.sum() > 0 and y_bin_f.sum() < len(y_bin_f):
        fpr_f, tpr_f, _ = roc_curve(y_bin_f, p_f)
        j_f = float(np.max(tpr_f - fpr_f))
        fold_Js.append(j_f)
        print(f"  fold {fold_i:2d}: J={j_f:.3f}  n_te={len(yte)}  HIGH_te={y_bin_f.sum()}")

fpr, tpr, thr_vals = roc_curve(y_bin, pool_p_notch)
j_scores = tpr - fpr
best_idx = int(np.argmax(j_scores))
NOTCH_THR = float(thr_vals[best_idx])
NOTCH_J   = float(j_scores[best_idx])

# Pipeline wrapper so predict_proba works the same as deployed model
class _NotchedPipeline:
    """Wraps StandardScaler + SVC to match sklearn Pipeline.predict_proba interface."""
    def predict_proba(self, X_sel: np.ndarray) -> np.ndarray:
        X_sc = NOTCH_SC.transform(X_sel)
        return NOTCH_SVC.predict_proba(X_sc)

_notched_pipeline = _NotchedPipeline()

print(f"\n  Notched 10-fold CV threshold : {NOTCH_THR:.4f}  (J={NOTCH_J:.3f})")
print(f"  Mean fold J                  : {np.mean(fold_Js):.3f}  "
      f"min={min(fold_Js):.3f}  max={max(fold_Js):.3f}")

# ---------------------------------------------------------------------------
# Step 4 — Variant C: windowed inference on control / adaptation (notched)
# ---------------------------------------------------------------------------
print("\n--- Step 4: Variant C — windowed inference on notched control/adaptation ---")

for cond in ("control", "adaptation"):
    print(f"  Loading {cond} (notched) ...", flush=True)
    data, ts = load_eeg(FILES[cond], notch25=True)
    p, t = windowed_phigh(data, ts, NOTCH_SEL, _notched_pipeline, NOTCH_MEAN, NOTCH_STD)
    var_p.setdefault("C", {})[cond] = p
    var_t.setdefault("C", {})[cond] = t
    print(f"    n_windows={len(p)}  mean={p.mean():.3f}  std={p.std():.3f}  "
          f"% > notched CV thr ({NOTCH_THR:.3f}): {(p > NOTCH_THR).mean():.1%}")

# ---------------------------------------------------------------------------
# Step 5 — Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY TABLE")
print("=" * 72)

THRESHOLDS = {
    "A": ("Deployed LORO",       DEPLOYED_LORO_THR),
    "B": ("Unnotched 10-fold CV", KFOLD_THR_UNNOTCHED),
    "C": (f"Notched 10-fold CV", NOTCH_THR),
}

print(f"\n{'Variant':<8} {'Threshold method':<28} {'Thr':>7}  "
      f"{'Ctrl % > thr':>13}  {'Adap % > thr':>13}  {'Ctrl mean':>9}  {'Adap mean':>9}")
print("-" * 95)
for var_id, (label, thr) in THRESHOLDS.items():
    p_ctrl = var_p[var_id]["control"]
    p_adap = var_p[var_id]["adaptation"]
    pct_c  = (p_ctrl > thr).mean()
    pct_a  = (p_adap > thr).mean()
    print(f"{var_id:<8} {label:<28} {thr:7.4f}  "
          f"{pct_c:>12.1%}  {pct_a:>12.1%}  "
          f"{p_ctrl.mean():>9.3f}  {p_adap.mean():>9.3f}")

# FM_Beta z-score comparison
if fm_beta_idx is not None:
    print(f"\nFM_Beta diagnostic:")
    print(f"  deployed std : {DEPLOYED_STD[fm_beta_idx]:.5f}")
    print(f"  notched  std : {NOTCH_STD[fm_beta_idx]:.5f}")
    # Show mean z-score for each condition in each regime
    for cond in ("control", "adaptation"):
        data_std, ts_std = load_eeg(FILES[cond], notch25=False)
        data_ntc, ts_ntc = load_eeg(FILES[cond], notch25=True)
        # Extract single window near the middle for quick comparison
        mid = data_std.shape[1] // 2
        win = int(WIN_S * SRATE)
        ep_std = data_std[:, mid - win : mid][np.newaxis, :, :]
        ep_ntc = data_ntc[:, mid - win : mid][np.newaxis, :, :]
        x_std, _ = _extract_feat(ep_std, SRATE, region_map)
        x_ntc, _ = _extract_feat(ep_ntc, SRATE, region_map)
        z_std = (x_std[0, fm_beta_idx] - DEPLOYED_MEAN[fm_beta_idx]) / DEPLOYED_STD[fm_beta_idx]
        z_ntc = (x_ntc[0, fm_beta_idx] - NOTCH_MEAN[fm_beta_idx])   / NOTCH_STD[fm_beta_idx]
        print(f"  {cond:<12}  FM_Beta (mid-window)  "
              f"z_deployed={z_std:+.1f}  z_notched={z_ntc:+.1f}")

# ---------------------------------------------------------------------------
# Step 6 — Plot
# ---------------------------------------------------------------------------
print("\n--- Step 6: Generating figure ---")

COLORS_VAR = {
    "A": ("#c0392b", DEPLOYED_LORO_THR,       "A: deployed LORO thr"),
    "B": ("#e67e22", KFOLD_THR_UNNOTCHED,      "B: 10-fold CV thr (unnotched)"),
    "C": ("#2980b9", NOTCH_THR,               "C: 10-fold CV thr (25 Hz notched)"),
}

fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
fig.suptitle("PDRY06 — 25 Hz Artifact Removal Simulation", fontsize=14, fontweight="bold")

COND_TITLES = {"control": "Control session", "adaptation": "Adaptation session"}

for row, cond in enumerate(("control", "adaptation")):
    ax_ts  = axes[row, 0]
    ax_bar = axes[row, 1]

    for var_id, (color, thr, label) in COLORS_VAR.items():
        p = var_p[var_id][cond]
        t = var_t[var_id][cond]
        t_rel = t - t[0]   # relative seconds
        ax_ts.plot(t_rel, p, color=color, alpha=0.8,
                   lw=1.5 if var_id == "C" else 1.0,
                   label=f"{label}  [thr={thr:.3f}]")
        ax_ts.axhline(thr, color=color, ls="--", lw=0.8, alpha=0.5)

    ax_ts.set_ylim(-0.05, 1.05)
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("P(HIGH)")
    ax_ts.set_title(f"{COND_TITLES[cond]} — p_high time series")
    ax_ts.legend(fontsize=7, loc="upper right")

    # Bar: % above threshold
    pcts = []
    bar_labels = []
    bar_colors = []
    for var_id, (color, thr, label) in COLORS_VAR.items():
        p = var_p[var_id][cond]
        pcts.append((p > thr).mean() * 100)
        bar_labels.append(f"Var {var_id}")
        bar_colors.append(color)

    ax_bar.bar(bar_labels, pcts, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("% windows above threshold")
    ax_bar.set_title(f"{COND_TITLES[cond]} — % HIGH")
    for xi, v in enumerate(pcts):
        ax_bar.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=9)

plt.savefig(OUT_FIG, dpi=150)
print(f"Saved: {OUT_FIG}")
print("\nDONE.")
