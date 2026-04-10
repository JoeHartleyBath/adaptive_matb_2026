"""Test LORO Youden J with per-run LOW-block normalisation for S006.

Hypothesis: the resting baseline was recorded during electrode settling (RMS
4× lower than C2), so norm_stats.json Z-scores are badly miscalibrated.
Per-run LOW-block norms remove the global amplitude offset between C1 and C2
before any feature selection or classification.

Compares three normalisation strategies:
  A) Global rest (current model)     — norm from rest_physio.xdf
  B) Global LOW  (current fallback)  — norm from pooled LOW blocks C1+C2
  C) Per-run LOW (proposed)          — C1 normed by C1-LOW, C2 normed by C2-LOW

For each strategy runs the full LORO CV (same params as calibrate.py:
SelectKBest k=35, SVC linear C=1.0) and reports:
  - fold J scores
  - pooled J + threshold
  - AUC per fold
  - mean P(HIGH)|HIGH  vs  P(HIGH)|LOW

Run:
    $env:MATB_SCENARIO_OFFSET_S = "0.943"
    .venv\\Scripts\\Activate.ps1
    python scripts/_tmp_loro_perrun_norm_s006.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod
from build_mwl_training_dataset import PREPROCESSING_CONFIG, WINDOW_CONFIG
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config — must match calibrate_participant.py
# ---------------------------------------------------------------------------
SRATE  = 128.0
CAL_K  = 35
CAL_C  = 1.0
SEED   = 42

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"
REST_XDF  = PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-rest_physio.xdf"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
]
TAGS = ["C1", "C2"]

LOW_LBL  = LABEL_MAP["LOW"]
HIGH_LBL = LABEL_MAP["HIGH"]

# ---------------------------------------------------------------------------
# Load raw features per XDF (un-normalised)
# ---------------------------------------------------------------------------
print("Loading XDFs ...\n")
region_map = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []

# xdf_raw_X[i]      : (n_windows, n_features) raw features
# xdf_y[i]          : (n_windows,) 3-class labels
# xdf_low_epochs[i] : (n_low_windows, n_channels, n_samples) for norm compute
xdf_raw_X: list[np.ndarray] = []
xdf_y: list[np.ndarray] = []
xdf_low_X: list[np.ndarray] = []   # raw features of LOW windows only

for xdf_path, tag in zip(CAL_XDFS, TAGS):
    print(f"  {tag}: ", end="", flush=True)
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        print("FAILED"); sys.exit(1)

    all_epochs: list[np.ndarray] = []
    all_labels: list[int] = []
    low_epochs: list[np.ndarray] = []

    for epochs, level_str in results:
        lbl = LABEL_MAP[level_str]
        X_raw, names = _extract_feat(epochs, SRATE, region_map)
        if not feat_names:
            feat_names = names
        all_epochs.append(X_raw)
        all_labels.extend([lbl] * len(X_raw))
        if lbl == LOW_LBL:
            low_epochs.append(X_raw)

    X_all = np.concatenate(all_epochs)
    y_all = np.array(all_labels, dtype=np.int64)
    X_low = np.concatenate(low_epochs) if low_epochs else X_all

    xdf_raw_X.append(X_all)
    xdf_y.append(y_all)
    xdf_low_X.append(X_low)
    print(f"n={len(y_all)}  LOW={int((y_all==LOW_LBL).sum())}  HIGH={int((y_all==HIGH_LBL).sum())}")

n_feat = len(feat_names)


# ---------------------------------------------------------------------------
# Helper: compute norm stats from a raw feature array
# ---------------------------------------------------------------------------
def _norm_stats(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_raw.mean(axis=0)
    std  = X_raw.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# Helper: run full LORO CV given a list of (X_norm, y) per XDF
# ---------------------------------------------------------------------------
def _run_loro(xdf_X_norm: list[np.ndarray],
              xdf_labels: list[np.ndarray],
              label: str) -> None:
    """Run 2-fold LORO, print per-fold and pooled diagnostics."""
    high_lbl = HIGH_LBL
    low_lbl  = LOW_LBL
    fold_j: list[float] = []
    pool_p: list[np.ndarray] = []
    pool_y: list[np.ndarray] = []

    print(f"\n  Strategy: {label}")
    for fold_i in range(2):
        train_X = np.concatenate([xdf_X_norm[j] for j in range(2) if j != fold_i])
        train_y = np.concatenate([xdf_labels[j]  for j in range(2) if j != fold_i])
        test_X  = xdf_X_norm[fold_i]
        test_y  = xdf_labels[fold_i]

        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            print(f"    fold {fold_i}: skipped (<2 classes)")
            continue

        k   = min(CAL_K, train_X.shape[1])
        sel = SelectKBest(f_classif, k=k)
        tr  = sel.fit_transform(train_X, train_y)
        sc  = StandardScaler()
        tr  = sc.fit_transform(tr)
        svc = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                  probability=True, random_state=SEED)
        svc.fit(tr, train_y)

        te = sc.transform(sel.transform(test_X))
        p_high = svc.predict_proba(te)[:, -1]

        y_bin = (test_y == test_y.max()).astype(int)
        auc   = roc_auc_score(y_bin, p_high)
        fpr_f, tpr_f, _ = roc_curve(y_bin, p_high)
        j_f   = float(np.max(tpr_f - fpr_f))

        p_h_given_HIGH = float(p_high[test_y == high_lbl].mean()) if (test_y == high_lbl).any() else float("nan")
        p_h_given_LOW  = float(p_high[test_y == low_lbl].mean())  if (test_y == low_lbl).any()  else float("nan")
        inv = "INVERTED" if p_h_given_HIGH < p_h_given_LOW else "ok"

        tag = TAGS[fold_i]
        print(f"    fold {fold_i} (test={tag}):  J={j_f:.4f}  AUC={auc:.4f}  "
              f"P(H|HIGH)={p_h_given_HIGH:.3f}  P(H|LOW)={p_h_given_LOW:.3f}  {inv}")

        fold_j.append(j_f)
        pool_p.append(p_high)
        pool_y.append(y_bin)

    if pool_p:
        all_p = np.concatenate(pool_p)
        all_y = np.concatenate(pool_y)
        if len(np.unique(all_y)) >= 2:
            fpr, tpr, thr = roc_curve(all_y, all_p)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            pooled_j   = float(j_scores[best_idx])
            pooled_thr = float(thr[best_idx])
        else:
            pooled_j = pooled_thr = float("nan")
        print(f"    pooled:     J={pooled_j:.4f}  threshold={pooled_thr:.4f}  "
              f"({'PASS ✓' if pooled_j >= 0.10 else 'FAIL  (< 0.10)'})")

    print()


# ---------------------------------------------------------------------------
# Strategy A — Global rest (exact current model)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("NORM STRATEGY COMPARISON")
print("=" * 70)

# Load rest norm stats written by the last calibration run
_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
rest_mean = np.array(_ns["mean"])
rest_std  = np.array(_ns["std"])
rest_std[rest_std < 1e-12] = 1.0

xdf_A = [(X - rest_mean) / rest_std for X in xdf_raw_X]
_run_loro(xdf_A, xdf_y, "A — global rest baseline (norm_stats.json)")

# ---------------------------------------------------------------------------
# Strategy B — Global LOW (pooled C1-LOW + C2-LOW)
# ---------------------------------------------------------------------------
low_all = np.concatenate(xdf_low_X)
glob_mean, glob_std = _norm_stats(low_all)

xdf_B = [(X - glob_mean) / glob_std for X in xdf_raw_X]
_run_loro(xdf_B, xdf_y, "B — global LOW (pooled C1-LOW + C2-LOW)")

# ---------------------------------------------------------------------------
# Strategy C — Per-run LOW (C1 normed by C1-LOW, C2 by C2-LOW)
# ---------------------------------------------------------------------------
per_run_norms = [_norm_stats(X_low) for X_low in xdf_low_X]
for (mn, st), tag in zip(per_run_norms, TAGS):
    print(f"  Per-run norm {tag}: mean_RMS={float(np.sqrt(np.mean(mn**2))):.4f}  "
          f"std_mean={float(st.mean()):.4f}")
print()

xdf_C = [(X - mn) / st for X, (mn, st) in zip(xdf_raw_X, per_run_norms)]
_run_loro(xdf_C, xdf_y, "C — per-run LOW  (C1 → C1-LOW norm, C2 → C2-LOW norm)")

# ---------------------------------------------------------------------------
# Extra: show discrimination (HIGH-LOW) in Z-space per strategy, for top
# sign-flipping features, to confirm the amplitude offset is removed
# ---------------------------------------------------------------------------
FLIP_FEATURES = ["FM_Beta", "Cen_Beta", "Par_HjAct", "Cen_Engagement",
                 "Cen_SpEnt", "Par_SpEnt", "Occ_SpEnt"]

print("=" * 70)
print("GLOBAL AMPLITUDE OFFSET (C2_mean - C1_mean, Z-normed) for key features")
print("  A large Δ_global means the baseline shift dominates over workload signal")
print(f"  {'Feature':<25}  {'A (rest)':>10}  {'B (glob_LOW)':>12}  {'C (per_low)':>12}")
print("  " + "-" * 65)

for fname in FLIP_FEATURES:
    fi = feat_names.index(fname) if fname in feat_names else None
    if fi is None:
        continue
    for strat_label, xdf_list in [("A", xdf_A), ("B", xdf_B), ("C", xdf_C)]:
        pass  # collect below

    deltas = {}
    for sl, xl in [("A", xdf_A), ("B", xdf_B), ("C", xdf_C)]:
        d = float(xl[1][:, fi].mean() - xl[0][:, fi].mean())  # C2 mean - C1 mean
        deltas[sl] = d

    print(f"  {fname:<25}  {deltas['A']:+10.3f}  {deltas['B']:+12.3f}  {deltas['C']:+12.3f}")

print()
print("Ideal: Δ_global ≈ 0 for all levels (amplitude offset removed)")
print("--- End ---")
