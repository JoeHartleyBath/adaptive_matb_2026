"""Quick test: 10-fold stratified CV threshold on C1+C2 (per-run LOW norm).

Run:
    $env:MATB_SCENARIO_OFFSET_S = "0.943"
    .venv\\Scripts\\python.exe scripts/_tmp_10fold_threshold_s006.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

SRATE = 128.0; CAL_K = 35; CAL_C = 1.0; SEED = 42
PHYSIO   = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
FEAT_CFG = _REPO / "config" / "eeg_feature_extraction.yaml"
META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]
CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
]
TAGS = ["C1", "C2"]
LOW_LBL = LABEL_MAP["LOW"]; HIGH_LBL = LABEL_MAP["HIGH"]

# ---------------------------------------------------------------------------
# Load + per-run LOW norm
# ---------------------------------------------------------------------------
region_map = _build_region_map(FEAT_CFG, CH_NAMES)
xdf_raw_X, xdf_y, xdf_low_X = [], [], []

for xdf_path, tag in zip(CAL_XDFS, TAGS):
    print(f"  {tag}: ", end="", flush=True)
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        sys.exit(1)
    all_feats, all_labels, low_feats = [], [], []
    for epochs, level_str in results:
        lbl = LABEL_MAP[level_str]
        X_raw, _ = _extract_feat(epochs, SRATE, region_map)
        all_feats.append(X_raw)
        all_labels.extend([lbl] * len(X_raw))
        if lbl == LOW_LBL:
            low_feats.append(X_raw)
    xdf_raw_X.append(np.concatenate(all_feats))
    xdf_y.append(np.array(all_labels, dtype=np.int64))
    xdf_low_X.append(np.concatenate(low_feats) if low_feats else np.concatenate(all_feats))

def _norm(X):
    m = X.mean(0); s = X.std(0); s[s < 1e-12] = 1.0; return m, s

norms = [_norm(Xl) for Xl in xdf_low_X]
X_all = np.concatenate([(X - mn) / st for X, (mn, st) in zip(xdf_raw_X, norms)])
y_all = np.concatenate(xdf_y)
# Run labels (for stratification within runs)
run_labels = np.concatenate([np.full(len(y), i) for i, y in enumerate(xdf_y)])

print(f"\nCombined: n={len(y_all)}  LOW={(y_all==LOW_LBL).sum()}  HIGH={(y_all==HIGH_LBL).sum()}")

# ---------------------------------------------------------------------------
# 10-fold stratified CV — pool all held-out predictions
# ---------------------------------------------------------------------------
print("\nRunning 10-fold stratified CV ...")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
pool_p = np.zeros(len(y_all))

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_all, y_all)):
    Xtr = X_all[tr_idx]; ytr = y_all[tr_idx]
    Xte = X_all[te_idx]; yte = y_all[te_idx]

    sel = SelectKBest(f_classif, k=min(CAL_K, Xtr.shape[1]))
    Xtr_s = sel.fit_transform(Xtr, ytr)
    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_s)
    svc   = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
    svc.fit(Xtr_s, ytr)

    Xte_s = sc.transform(sel.transform(Xte))
    p_high = svc.predict_proba(Xte_s)[:, -1]
    pool_p[te_idx] = p_high

    y_bin_fold = (yte == HIGH_LBL).astype(int)
    if y_bin_fold.sum() > 0 and y_bin_fold.sum() < len(y_bin_fold):
        fpr_f, tpr_f, _ = roc_curve(y_bin_fold, p_high)
        j_f = float(np.max(tpr_f - fpr_f))
        auc_f = roc_auc_score(y_bin_fold, p_high)
        print(f"  fold {fold_i:2d}: J={j_f:.3f}  AUC={auc_f:.3f}  "
              f"runs={sorted(set(run_labels[te_idx]))}")

# Pooled ROC + threshold
y_bin = (y_all == HIGH_LBL).astype(int)
fpr, tpr, thr_arr = roc_curve(y_bin, pool_p)
j_arr = tpr - fpr
best_i = int(np.argmax(j_arr))

# Also compute minimum-distance-to-(0,1) threshold (as per the paper)
dist = np.sqrt(fpr**2 + (1 - tpr)**2)
min_dist_i = int(np.argmin(dist))

pooled_j   = float(j_arr[best_i])
pooled_thr = float(thr_arr[best_i])
md_thr     = float(thr_arr[min_dist_i])
auc_pool   = roc_auc_score(y_bin, pool_p)

print(f"\n{'='*55}")
print(f"  Pooled 10-fold  AUC={auc_pool:.3f}  J={pooled_j:.3f}")
print(f"  Youden threshold:           {pooled_thr:.4f}")
print(f"  Min-distance threshold:     {md_thr:.4f}")
print()

# Compare class means
for lname, lbl in [("LOW", LOW_LBL), ("MOD", LABEL_MAP["MODERATE"]), ("HIGH", HIGH_LBL)]:
    m = pool_p[y_all == lbl]
    print(f"  mean p_high | {lname}: {m.mean():.3f}  (median={np.median(m):.3f}  p10={np.percentile(m,10):.3f}  p90={np.percentile(m,90):.3f})")

# Compare against LORO and train-set thresholds
print(f"\n  LORO threshold (Strategy C):    0.0328")
print(f"  Training-set threshold:         0.2137")
print(f"  10-fold Youden threshold:       {pooled_thr:.4f}")
print(f"  10-fold min-distance threshold: {md_thr:.4f}")
print("--- End ---")
