"""Threshold diagnostic for PDRY06 — what should the threshold have been?

PDRY06 model_config.json (produced 2026-04-28, threshold_method="loro"):
    loro_fold_j_scores     : [0.448544, 0.412945]
    loro_youdens_j         : 0.249515  (pooled)
    loro_youden_threshold  : 0.484876
    train_youdens_j        : 0.656311
    train_youden_threshold : 0.333375

LORO worked (J=0.249 ≥ 0.10 gate) so LORO threshold was deployed.
This script computes what the 10-fold stratified CV threshold would have been
under the new calibrate_participant.py logic, and shows the LORO diagnostics
for comparison.

Phases:
    1  XDF block audit (exact _load_xdf_block path used by calibrate.py)
    2  10-fold stratified CV — threshold, J, per-fold breakdown
    3  LORO forensics — replicate original calibration (fold-by-fold detail)
    4  Feature direction analysis — sign agreement C1 vs C2

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_threshold_diagnostic_pdry06.py
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config — must match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
SRATE = 128.0
CAL_K = 35
CAL_C = 1.0
SEED  = 42
CV_N_SPLITS = 10

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

_ns       = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

_mc = json.loads((MODEL_DIR / "model_config.json").read_text())
print("=" * 72)
print("PDRY06 deployed model_config.json")
print("=" * 72)
print(f"  threshold_method        : {_mc['threshold_method']}")
print(f"  youden_threshold        : {_mc['youden_threshold']}")
print(f"  youdens_j               : {_mc['youdens_j']}")
print(f"  train_youden_threshold  : {_mc['train_youden_threshold']}")
print(f"  train_youdens_j         : {_mc['train_youdens_j']}")
print(f"  loro_youden_threshold   : {_mc.get('loro_youden_threshold')}")
print(f"  loro_youdens_j          : {_mc.get('loro_youdens_j')}")
print(f"  loro_fold_j_scores      : {_mc.get('loro_fold_j_scores')}")

CAL_XDFS = [
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf",
]

# ==================================================================================
# Phase 1 — XDF Block Audit
# ==================================================================================

print("\n")
print("=" * 72)
print("PHASE 1 — XDF BLOCK AUDIT")
print("=" * 72)
print(f"scenario offset: {_cal_mod._MATB_SCENARIO_OFFSET_S}s  "
      f"(env override: {os.environ.get('MATB_SCENARIO_OFFSET_S', 'none')})")

region_map = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []

xdf_X_norm_list: list[np.ndarray] = []
xdf_y_list:      list[np.ndarray] = []
run_raw:         list[dict]       = []

for xdf_idx, xdf_path in enumerate(CAL_XDFS):
    acq_tag = "C1" if "c1" in xdf_path.stem else "C2"
    print(f"\n[XDF {xdf_idx}] {acq_tag}  {xdf_path.name}")

    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        print("  _load_xdf_block returned None — SKIPPED")
        continue

    print(f"  _load_xdf_block returned {len(results)} (epochs, level) pairs")
    label_counts: dict[str, int] = {}
    class_feats:  dict[int, list] = {}
    xdf_y_raw:    list[int] = []
    xdf_epochs_all: list[np.ndarray] = []

    for epochs, level in results:
        n_win = epochs.shape[0]
        label_counts[level] = label_counts.get(level, 0) + n_win
        print(f"  level={level:<10}  n_windows={n_win}")
        lbl = LABEL_MAP[level]
        X_block, names = _extract_feat(epochs, SRATE, region_map)
        if not feat_names:
            feat_names = names
        class_feats.setdefault(lbl, []).append(X_block)
        xdf_y_raw.extend([lbl] * n_win)
        xdf_epochs_all.append(epochs)

    y_arr     = np.array(xdf_y_raw, dtype=np.int64)
    high_lbl  = LABEL_MAP["HIGH"]
    has_high  = (y_arr == high_lbl).any()
    print(f"\n  Summary: {dict(sorted(label_counts.items()))}")
    print(f"  HIGH windows: {int((y_arr == high_lbl).sum())}  "
          f"MOD: {int((y_arr == LABEL_MAP.get('MODERATE', 1)).sum())}  "
          f"LOW: {int((y_arr == LABEL_MAP['LOW']).sum())}  "
          f"HIGH present: {has_high}")

    all_eps = np.concatenate(xdf_epochs_all)
    X_all, _ = _extract_feat(all_eps, SRATE, region_map)
    X_norm   = (X_all - NORM_MEAN) / NORM_STD
    xdf_X_norm_list.append(X_norm)
    xdf_y_list.append(y_arr)
    run_raw.append({k: np.concatenate(v) for k, v in class_feats.items()})

if len(xdf_X_norm_list) < 2:
    print("ERROR: fewer than 2 XDFs loaded")
    sys.exit(1)

cal_X_norm = np.concatenate(xdf_X_norm_list)
cal_y      = np.concatenate(xdf_y_list)
print(f"\nCombined calibration: n={len(cal_y)}  "
      f"LOW={(cal_y==LABEL_MAP['LOW']).sum()}  "
      f"MOD={(cal_y==LABEL_MAP.get('MODERATE',1)).sum()}  "
      f"HIGH={(cal_y==LABEL_MAP['HIGH']).sum()}")

# ==================================================================================
# Phase 2 — 10-fold stratified CV (new calibrate_participant.py logic)
# ==================================================================================

print("\n")
print("=" * 72)
print(f"PHASE 2 — {CV_N_SPLITS}-FOLD STRATIFIED CV THRESHOLD (new logic)")
print("=" * 72)

high_label = LABEL_MAP["HIGH"]
y_binary   = (cal_y == high_label).astype(int)

cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=SEED)
pool_p_kfold = np.zeros(len(cal_y))

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(cal_X_norm, cal_y)):
    Xtr = cal_X_norm[tr_idx]; ytr = cal_y[tr_idx]
    Xte = cal_X_norm[te_idx]; yte = cal_y[te_idx]

    k   = min(CAL_K, Xtr.shape[1])
    sel = SelectKBest(f_classif, k=k)
    Xtr_s = sel.fit_transform(Xtr, ytr)
    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_s)
    svc   = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
    svc.fit(Xtr_s, ytr)

    Xte_s = sc.transform(sel.transform(Xte))
    p_f   = svc.predict_proba(Xte_s)[:, -1]
    pool_p_kfold[te_idx] = p_f

    y_bin_f = (yte == high_label).astype(int)
    if y_bin_f.sum() > 0 and y_bin_f.sum() < len(y_bin_f):
        fpr_f, tpr_f, _ = roc_curve(y_bin_f, p_f)
        j_f   = float(np.max(tpr_f - fpr_f))
        auc_f = roc_auc_score(y_bin_f, p_f)
        print(f"  fold {fold_i:2d}: J={j_f:.3f}  AUC={auc_f:.3f}  "
              f"n_te={len(yte)}  HIGH_te={y_bin_f.sum()}")
    else:
        print(f"  fold {fold_i:2d}: single class in test — J=N/A")

fpr_k, tpr_k, thr_k = roc_curve(y_binary, pool_p_kfold)
j_k      = tpr_k - fpr_k
best_k   = int(np.argmax(j_k))
kfold_j  = float(j_k[best_k])
kfold_thr = float(thr_k[best_k])
kfold_auc = roc_auc_score(y_binary, pool_p_kfold)

print(f"\n  Pooled {CV_N_SPLITS}-fold  AUC={kfold_auc:.3f}  J={kfold_j:.3f}")
print(f"  10-fold Youden threshold:  {kfold_thr:.4f}")

for lname, lbl in [("LOW", LABEL_MAP["LOW"]),
                   ("MOD", LABEL_MAP.get("MODERATE", 1)),
                   ("HIGH", high_label)]:
    m = pool_p_kfold[cal_y == lbl]
    if len(m):
        print(f"  mean p_high | {lname}: {m.mean():.3f}  "
              f"(median={np.median(m):.3f}  "
              f"p10={np.percentile(m,10):.3f}  p90={np.percentile(m,90):.3f})")

# ==================================================================================
# Phase 3 — LORO forensics (what actually ran on 2026-04-28)
# ==================================================================================

print("\n")
print("=" * 72)
print("PHASE 3 — LORO FORENSICS (what calibrate.py ran on 2026-04-28)")
print("=" * 72)

FOLD_NAMES = {0: "Fold 0  (train=C2, test=C1)", 1: "Fold 1  (train=C1, test=C2)"}
pool_p_loro: list[np.ndarray] = []
pool_y_loro: list[np.ndarray] = []

for fold_i in range(2):
    print(f"\n--- {FOLD_NAMES[fold_i]} ---")
    train_X = np.concatenate([xdf_X_norm_list[j] for j in range(2) if j != fold_i])
    train_y = np.concatenate([xdf_y_list[j]      for j in range(2) if j != fold_i])
    test_X  = xdf_X_norm_list[fold_i]
    test_y  = xdf_y_list[fold_i]

    for lbl_name, lbl_val in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        n_tr = int((train_y == lbl_val).sum())
        n_te = int((test_y  == lbl_val).sum())
        print(f"  {lbl_name:<10} ({lbl_val}):  train={n_tr:4d}  test={n_te:4d}")

    k   = min(CAL_K, train_X.shape[1])
    sel = SelectKBest(f_classif, k=k)
    train_X_sel = sel.fit_transform(train_X, train_y)
    sc  = StandardScaler()
    train_X_sc  = sc.fit_transform(train_X_sel)
    svc = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
              probability=True, random_state=SEED)
    svc.fit(train_X_sc, train_y)

    test_X_sc   = sc.transform(sel.transform(test_X))
    p_high_fold = svc.predict_proba(test_X_sc)[:, -1]

    y_bin_fold = (test_y == test_y.max()).astype(int)
    print(f"  test_y.max() = {int(test_y.max())}  "
          f"(should be {high_label}  "
          f"→ {'OK' if test_y.max() == high_label else '>>> MISMATCH <<<'})")

    if y_bin_fold.sum() > 0 and y_bin_fold.sum() < len(y_bin_fold):
        auc = roc_auc_score(y_bin_fold, p_high_fold)
        fpr_f, tpr_f, _ = roc_curve(y_bin_fold, p_high_fold)
        j_f = float(np.max(tpr_f - fpr_f))
        print(f"  AUC={auc:.4f}  Fold J={j_f:.4f}")
    else:
        print(f"  single class in test — AUC/J N/A")

    for lbl_name, lbl_val in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        mask = (test_y == lbl_val)
        if mask.any():
            print(f"  mean P(HIGH) | {lbl_name:<10} = {float(p_high_fold[mask].mean()):.4f}")

    pool_p_loro.append(p_high_fold)
    pool_y_loro.append(y_bin_fold)

print("\n--- LORO Pooled ---")
all_p_l = np.concatenate(pool_p_loro)
all_y_l = np.concatenate(pool_y_loro)
fpr_l, tpr_l, thr_l = roc_curve(all_y_l, all_p_l)
j_l      = tpr_l - fpr_l
best_l   = int(np.argmax(j_l))
print(f"  pooled J={float(j_l[best_l]):.4f}  threshold={float(thr_l[best_l]):.4f}")

# ==================================================================================
# Phase 4 — Feature direction analysis
# ==================================================================================

print("\n")
print("=" * 72)
print("PHASE 4 — FEATURE DIRECTION ANALYSIS (sign agreement C1 vs C2)")
print("=" * 72)

n_feat  = len(feat_names)
low_lbl_v = LABEL_MAP["LOW"]
high_lbl_v = LABEL_MAP["HIGH"]
mod_lbl_v  = LABEL_MAP.get("MODERATE", 1)
means_z    = np.full((2, 3, n_feat), np.nan)

for r_idx, cf in enumerate(run_raw):
    for lbl, X_raw in cf.items():
        X_z = (X_raw - NORM_MEAN) / NORM_STD
        ci  = {low_lbl_v: 0, mod_lbl_v: 1, high_lbl_v: 2}.get(lbl)
        if ci is not None:
            means_z[r_idx, ci] = X_z.mean(axis=0)

disc_c1 = means_z[0, 2] - means_z[0, 0]
disc_c2 = means_z[1, 2] - means_z[1, 0]
valid   = np.isfinite(disc_c1) & np.isfinite(disc_c2)
agrees  = np.sign(disc_c1) == np.sign(disc_c2)
n_valid = int(valid.sum())
n_flip  = int((valid & ~agrees).sum())

print(f"\nValid features (both runs): {n_valid} / {n_feat}")
print(f"  Sign agrees : {int((valid & agrees).sum())}")
print(f"  Sign FLIPS  : {n_flip}  ({n_flip/max(n_valid,1):.1%})")

def _binary_Xy(cf, lo, hi):
    parts_X, parts_y = [], []
    for lbl in (lo, hi):
        if lbl in cf:
            X   = cf[lbl]
            X_z = (X - NORM_MEAN) / NORM_STD
            parts_X.append(X_z)
            parts_y.append(np.full(len(X_z), int(lbl == hi), dtype=int))
    if len(parts_X) < 2:
        return None, None
    return np.concatenate(parts_X), np.concatenate(parts_y)

X_c1, y_c1 = _binary_Xy(run_raw[0], low_lbl_v, high_lbl_v)
X_c2, y_c2 = _binary_Xy(run_raw[1], low_lbl_v, high_lbl_v)
top35_c1, top35_c2 = set(), set()

for tag, X, y, top_set in [("C1", X_c1, y_c1, top35_c1),
                             ("C2", X_c2, y_c2, top35_c2)]:
    if X is not None and len(np.unique(y)) == 2:
        fsel = SelectKBest(f_classif, k=min(CAL_K, n_feat))
        fsel.fit(X, y)
        top_set.update(np.argsort(fsel.scores_)[::-1][:CAL_K].tolist())

if top35_c1 and top35_c2:
    overlap = top35_c1 & top35_c2
    print(f"\nTop-{CAL_K} feature overlap (C1 ∩ C2): {len(overlap)}/{CAL_K}  "
          f"({len(overlap)/CAL_K:.0%})")

combined_top = top35_c1 | top35_c2
flip_top = [(i, feat_names[i], float(disc_c1[i]), float(disc_c2[i]))
            for i in combined_top
            if i < n_feat and np.isfinite(disc_c1[i]) and np.isfinite(disc_c2[i])
            and np.sign(disc_c1[i]) != np.sign(disc_c2[i])]
flip_top.sort(key=lambda x: abs(x[2]) + abs(x[3]), reverse=True)

print(f"\nTop-{CAL_K}-candidate features with sign flip (n={len(flip_top)}):")
print(f"  {'idx':>4}  {'feature':<40}  {'disc_C1':>9}  {'disc_C2':>9}")
for idx, name, d1, d2 in flip_top[:20]:
    print(f"  {idx:4d}  {name:<40}  {d1:+.4f}    {d2:+.4f}")

# ==================================================================================
# Summary
# ==================================================================================

print("\n")
print("=" * 72)
print("SUMMARY — threshold comparison")
print("=" * 72)
print(f"  Deployed on 2026-04-28 (LORO)  :  {_mc['youden_threshold']:.4f}  "
      f"(J={_mc['youdens_j']:.4f})")
print(f"  Train-set                       :  {_mc['train_youden_threshold']:.4f}  "
      f"(J={_mc['train_youdens_j']:.4f})")
print(f"  10-fold CV (new logic)          :  {kfold_thr:.4f}  "
      f"(J={kfold_j:.4f}  AUC={kfold_auc:.4f})")
print()
print(f"  Offline adaptation % above threshold:")
print(f"    with LORO threshold  ({_mc['youden_threshold']:.4f}): "
      f"see results from _tmp_verify_phigh_drift_pdry06.py  (14.2%)")
print(f"    with 10-fold threshold ({kfold_thr:.4f}): would be "
      f"~{100 * float(np.mean(pool_p_kfold > kfold_thr)):.1f}%  "
      f"(on cal data — not adaptation XDF)")
print("\n--- End ---")
