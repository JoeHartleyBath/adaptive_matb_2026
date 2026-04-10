"""Diagnostic: why did LORO Youden's J fail for PSELF S006?

model_config.json recorded:
    loro_fold_j_scores : [0.0, 0.076052]
    loro_youdens_j     : 0.0  (pooled)
    loro_youden_threshold : Infinity
    threshold_method   : train_set_fallback

Fold directions (sorted XDF order):
    Fold 0 — train=C2, test=C1 → J=0.0   ← TOTAL FAILURE
    Fold 1 — train=C1, test=C2 → J=0.076 ← weak but positive

This script runs three diagnostic phases:

Phase 1 — XDF Block Audit
    For each calibration XDF: list every loaded block, counts per workload
    level, and check whether HIGH windows exist at all.

Phase 2 — Per-fold prediction forensics
    Replicate _compute_loro_threshold() with added per-fold checks:
    • test_y.max() should be LABEL_MAP["HIGH"]=2 (H1: silent mislabelling)
    • np.isfinite(p_high_fold).all() (H4: Inf probabilities)
    • AUC per fold (< 0.5 → systematic inversion → H2)
    • Class-conditional mean P(HIGH): if P(HIGH)|HIGH < P(HIGH)|LOW → inversion

Phase 3 — Per-feature cross-run direction analysis
    For each of the 54 features:
    • Z-normed (HIGH_mean − LOW_mean) per run
    • Sign agreement / disagreement between C1 and C2
    • SelectKBest F-scores per run separately; feature-set overlap at k=35

Run:
    .venv\\Scripts\\Activate.ps1
    python scripts/_tmp_loro_s006_diagnose.py
"""

from __future__ import annotations

import json
import os
import re
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

from build_mwl_training_dataset import (
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _extract_all_blocks,
    _find_stream,
    _merge_eeg_streams,
    _parse_markers,
)
import calibrate_participant as _cal_mod
from eeg import EegPreprocessor, extract_windows, slice_block
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config — must match calibrate_participant.py exactly
# ---------------------------------------------------------------------------
SRATE            = 128.0
CAL_K            = 35       # frozen as of 2026-04-10
CAL_C            = 1.0
SEED             = 42

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PSELF")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"

META      = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES  = META["channel_names"]

_ns        = json.loads((MODEL_DIR / "norm_stats.json").read_text())
NORM_MEAN  = np.array(_ns["mean"])
NORM_STD   = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0

# These are the two files that sorted alphabetically become XDF[0] and XDF[1]
# after the _old\d+ filter.  Confirms the XDF ordering in LORO.
CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",   # XDF[0] — fold 0 test set
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",   # XDF[1] — fold 1 test set
]

# ==================================================================================
# Helpers
# ==================================================================================

def _extract_epochs(pp, eeg_ts, t_start, t_end):
    i0 = int(np.searchsorted(eeg_ts, t_start))
    i1 = int(np.searchsorted(eeg_ts, t_end))
    block = slice_block(pp, i0, i1, WINDOW_CONFIG)
    return extract_windows(block, WINDOW_CONFIG)


# ==================================================================================
# Phase 1 — XDF Block Audit
# ==================================================================================

print("=" * 72)
print("PHASE 1 — XDF BLOCK AUDIT (via _load_xdf_block — exact match to calibrate.py)")
print("=" * 72)
print(f"scenario offset: {_cal_mod._MATB_SCENARIO_OFFSET_S}s  (env override: {os.environ.get('MATB_SCENARIO_OFFSET_S', 'none')})")

region_map  = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []

# Will hold per-XDF (Z-normed features, 3-class labels) exactly as calibrate.py does
xdf_X_norm_list: list[np.ndarray] = []
xdf_y_list: list[np.ndarray]      = []

# Also raw features for Phase 3
run_raw: list[dict[int, list]] = []

for xdf_idx, xdf_path in enumerate(CAL_XDFS):
    acq_tag = "C1" if "c1" in xdf_path.stem else "C2"
    print(f"\n[XDF {xdf_idx}] {acq_tag}  {xdf_path.name}")

    # Call the real _load_xdf_block — includes scenario fallback
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)

    if results is None:
        print("  _load_xdf_block returned None — would be SKIPPED in calibrate.py")
        print("  NOTE: if this XDF is in xdf_files_loaded, something is inconsistent")
        continue

    print(f"  _load_xdf_block returned {len(results)} (epochs, level) pairs")
    label_counts: dict[str, int] = {}
    class_feats: dict[int, list] = {}
    xdf_y_raw: list[int] = []
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

    y_arr = np.array(xdf_y_raw, dtype=np.int64)
    high_lbl = LABEL_MAP["HIGH"]
    has_high = (y_arr == high_lbl).any()
    print(f"\n  Summary: {dict(sorted(label_counts.items()))}")
    print(f"  HIGH windows present: {has_high}  "
          f"(n_HIGH={int((y_arr == high_lbl).sum())}, "
          f"n_MODERATE={int((y_arr == LABEL_MAP.get('MODERATE', 1)).sum())}, "
          f"n_LOW={int((y_arr == LABEL_MAP['LOW']).sum())})")

    if not has_high:
        print(f"  >>> H1 CANDIDATE: XDF {xdf_idx} ({acq_tag}) has NO HIGH windows <<<")

    # Build Z-normed feature matrix
    all_eps = np.concatenate(xdf_epochs_all)
    X_all, _ = _extract_feat(all_eps, SRATE, region_map)
    X_norm = (X_all - NORM_MEAN) / NORM_STD
    xdf_X_norm_list.append(X_norm)
    xdf_y_list.append(y_arr)
    run_raw.append({k: np.concatenate(v) for k, v in class_feats.items()})

# ==================================================================================
# Phase 2 — Per-fold LORO prediction forensics
# ==================================================================================

print("\n")
print("=" * 72)
print("PHASE 2 — PER-FOLD LORO PREDICTION FORENSICS")
print("=" * 72)

if len(xdf_X_norm_list) < 2:
    print("ERROR: fewer than 2 XDFs loaded — cannot run LORO")
    sys.exit(1)

high_label = LABEL_MAP["HIGH"]
pool_p: list[np.ndarray] = []
pool_y: list[np.ndarray] = []
fold_j: list[float]      = []

FOLD_NAMES = {0: "Fold 0  (train=C2, test=C1)", 1: "Fold 1  (train=C1, test=C2)"}

for fold_i in range(2):
    print(f"\n--- {FOLD_NAMES[fold_i]} ---")

    train_X = np.concatenate([xdf_X_norm_list[j] for j in range(2) if j != fold_i])
    train_y = np.concatenate([xdf_y_list[j]      for j in range(2) if j != fold_i])
    test_X  = xdf_X_norm_list[fold_i]
    test_y  = xdf_y_list[fold_i]

    # Class distribution check
    for lbl, name in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        n_train = int((train_y == lbl).sum())
        n_test  = int((test_y  == lbl).sum())
        print(f"  {name} ({lbl}):  train={n_train:4d}  test={n_test:4d}")

    train_unique = np.unique(train_y)
    test_unique  = np.unique(test_y)
    print(f"  train classes: {train_unique.tolist()}")
    print(f"  test  classes: {test_unique.tolist()}")

    if len(train_unique) < 2 or len(test_unique) < 2:
        print("  WOULD RETURN None, None, [] — early exit guard triggered in real LORO")
        continue

    # Fit fold model
    k = min(CAL_K, train_X.shape[1])
    sel = SelectKBest(f_classif, k=k)
    train_X_sel = sel.fit_transform(train_X, train_y)
    sc  = StandardScaler()
    train_X_sc  = sc.fit_transform(train_X_sel)
    svc = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
              probability=True, random_state=SEED)
    svc.fit(train_X_sc, train_y)

    print(f"  SVC trained  classes_: {svc.classes_.tolist()}")

    test_X_sc   = sc.transform(sel.transform(test_X))
    p_high_fold = svc.predict_proba(test_X_sc)[:, -1]

    # H4: non-finite probabilities?
    finite_ok = bool(np.isfinite(p_high_fold).all())
    print(f"  p_high finite: {finite_ok}  "
          f"min={float(np.nanmin(p_high_fold)):.4f}  "
          f"max={float(np.nanmax(p_high_fold)):.4f}")

    # Binary label as in _compute_loro_threshold
    y_bin_fold = (test_y == test_y.max()).astype(int)
    effective_pos_class = int(test_y.max())
    print(f"  test_y.max() = {effective_pos_class}  "
          f"(should be {high_label} for HIGH)  "
          f"→ {'OK' if effective_pos_class == high_label else '>>> H1: test_y.max() != HIGH <<<'}")

    n_pos = int(y_bin_fold.sum())
    n_neg = int((y_bin_fold == 0).sum())
    print(f"  y_bin_fold positives={n_pos}  negatives={n_neg}")

    # AUC
    if n_pos > 0 and n_neg > 0:
        auc = roc_auc_score(y_bin_fold, p_high_fold)
        print(f"  AUC={auc:.4f}  {'>>> H2: AUC < 0.5 → SYSTEMATIC INVERSION <<<' if auc < 0.5 else ''}")
    else:
        auc = float("nan")
        print("  AUC: cannot compute (single class in y_bin_fold)")

    # Class-conditional mean P(HIGH)
    for lbl_name, lbl_val in sorted(LABEL_MAP.items(), key=lambda kv: kv[1]):
        mask = (test_y == lbl_val)
        if mask.any():
            mean_p = float(p_high_fold[mask].mean())
            print(f"  mean P(HIGH) | {lbl_name:<10} = {mean_p:.4f}")

    # Inversion check
    mask_high = (test_y == high_label)
    mask_low  = (test_y == LABEL_MAP["LOW"])
    if mask_high.any() and mask_low.any():
        p_given_high = float(p_high_fold[mask_high].mean())
        p_given_low  = float(p_high_fold[mask_low].mean())
        if p_given_high < p_given_low:
            print(f"  >>> H2: INVERTED — P(HIGH|HIGH)={p_given_high:.4f} < P(HIGH|LOW)={p_given_low:.4f} <<<")
        else:
            print(f"  Direction OK: P(HIGH|HIGH)={p_given_high:.4f} > P(HIGH|LOW)={p_given_low:.4f}")

    fpr_f, tpr_f, _ = roc_curve(y_bin_fold, p_high_fold)
    j_f = float(np.max(tpr_f - fpr_f))
    print(f"  Fold J = {j_f:.6f}")
    fold_j.append(j_f)

    pool_p.append(p_high_fold)
    pool_y.append(y_bin_fold)

# Pooled ROC
print("\n--- Pooled ---")
all_p = np.concatenate(pool_p)
all_y = np.concatenate(pool_y)
print(f"  all_p finite: {bool(np.isfinite(all_p).all())}  "
      f"min={float(np.nanmin(all_p)):.4f}  max={float(np.nanmax(all_p)):.4f}")
print(f"  all_y positives={int(all_y.sum())}  negatives={int((all_y == 0).sum())}")

if len(np.unique(all_y)) >= 2:
    fpr, tpr, thr = roc_curve(all_y, all_p)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    print(f"  pooled J={float(j_scores[best_idx]):.6f}  "
          f"threshold={float(thr[best_idx]):.6f}")
    print(f"  thr[0]={float(thr[0]):.6f}  "
          f"(Infinity means argmax hits the first point → degenerate ROC)")
else:
    print("  Cannot compute pooled ROC (single class)")

# ==================================================================================
# Phase 3 — Per-feature cross-run direction analysis
# ==================================================================================

print("\n")
print("=" * 72)
print("PHASE 3 — PER-FEATURE CROSS-RUN DIRECTION ANALYSIS")
print("=" * 72)

if len(run_raw) < 2 or not feat_names:
    print("Not enough data for Phase 3")
    sys.exit(0)

n_feat = len(feat_names)
low_lbl  = LABEL_MAP["LOW"]
high_lbl = LABEL_MAP["HIGH"]
mod_lbl  = LABEL_MAP.get("MODERATE", 1)

# Z-normed means per run, per class
means_z = np.full((2, 3, n_feat), np.nan)  # (run, class, feat)

for r_idx, class_feats in enumerate(run_raw):
    for lbl, X_raw in class_feats.items():
        X_z = (X_raw - NORM_MEAN) / NORM_STD
        class_idx = {low_lbl: 0, mod_lbl: 1, high_lbl: 2}.get(lbl)
        if class_idx is not None:
            means_z[r_idx, class_idx] = X_z.mean(axis=0)

# H-L discrimination direction per run
disc_c1 = means_z[0, 2] - means_z[0, 0]  # HIGH - LOW, C1
disc_c2 = means_z[1, 2] - means_z[1, 0]  # HIGH - LOW, C2

valid    = np.isfinite(disc_c1) & np.isfinite(disc_c2)
agrees   = np.sign(disc_c1) == np.sign(disc_c2)

n_valid   = int(valid.sum())
n_agree   = int((valid & agrees).sum())
n_flip    = int((valid & ~agrees).sum())
flip_frac = n_flip / max(n_valid, 1)

print(f"\nFeatures with valid disc scores (both runs): {n_valid} / {n_feat}")
print(f"  Sign agrees (C1 and C2 discriminate same direction): {n_agree}")
print(f"  Sign FLIPS  (direction inverts between C1 and C2):    {n_flip}  "
      f"({flip_frac:.1%} of valid features)")

if flip_frac > 0.3:
    print("  >>> H2 CANDIDATE: >30% of features flip direction between runs <<<")

# SelectKBest F-scores per run separately
fsel_c1 = SelectKBest(f_classif, k=min(CAL_K, n_feat))
fsel_c2 = SelectKBest(f_classif, k=min(CAL_K, n_feat))

# Build binary data for F-score (LOW vs HIGH only)
def _binary_Xy(class_feats, low_lbl, high_lbl):
    parts_X, parts_y = [], []
    for lbl in (low_lbl, high_lbl):
        if lbl in class_feats:
            X = (np.concatenate(class_feats[lbl]) if isinstance(class_feats[lbl], list)
                 else class_feats[lbl])
            X_z = (X - NORM_MEAN) / NORM_STD
            parts_X.append(X_z)
            parts_y.append(np.full(len(X_z), lbl == high_lbl, dtype=int))
    if len(parts_X) < 2:
        return None, None
    return np.concatenate(parts_X), np.concatenate(parts_y)

X_c1, y_c1 = _binary_Xy(run_raw[0], low_lbl, high_lbl)
X_c2, y_c2 = _binary_Xy(run_raw[1], low_lbl, high_lbl)

top35_c1 = set()
top35_c2 = set()

if X_c1 is not None and len(np.unique(y_c1)) == 2:
    fsel_c1.fit(X_c1, y_c1)
    top35_c1 = set(np.argsort(fsel_c1.scores_)[::-1][:CAL_K].tolist())
    print(f"\nTop-{CAL_K} SelectKBest features in C1: {sorted(top35_c1)}")
else:
    print("\nC1: cannot compute SelectKBest (missing LOW or HIGH)")

if X_c2 is not None and len(np.unique(y_c2)) == 2:
    fsel_c2.fit(X_c2, y_c2)
    top35_c2 = set(np.argsort(fsel_c2.scores_)[::-1][:CAL_K].tolist())
    print(f"Top-{CAL_K} SelectKBest features in C2: {sorted(top35_c2)}")
else:
    print("C2: cannot compute SelectKBest (missing LOW or HIGH)")

if top35_c1 and top35_c2:
    overlap = top35_c1 & top35_c2
    print(f"\nFeature-set overlap (top-{CAL_K} C1 ∩ C2): "
          f"{len(overlap)} / {CAL_K}  ({len(overlap)/CAL_K:.0%})")
    if len(overlap) / CAL_K < 0.5:
        print("  >>> H2/feature instability: <50% of top features shared between runs <<<")

# Show features that flip direction AND are top-35 in at least one run
if top35_c1 or top35_c2:
    combined_top = top35_c1 | top35_c2
    flip_top = [(i, feat_names[i],
                  float(disc_c1[i]) if np.isfinite(disc_c1[i]) else float("nan"),
                  float(disc_c2[i]) if np.isfinite(disc_c2[i]) else float("nan"))
                for i in combined_top
                if i < n_feat and np.isfinite(disc_c1[i]) and np.isfinite(disc_c2[i])
                and np.sign(disc_c1[i]) != np.sign(disc_c2[i])]
    flip_top.sort(key=lambda x: abs(x[2]) + abs(x[3]), reverse=True)

    print(f"\nTop-35-candidate features with SIGN FLIP (n={len(flip_top)}):")
    print(f"  {'idx':>4}  {'feature':<40}  {'disc_C1':>9}  {'disc_C2':>9}")
    for idx, name, d1, d2 in flip_top[:20]:
        print(f"  {idx:4d}  {name:<40}  {d1:+.4f}    {d2:+.4f}")

print("\n--- End of diagnostics ---")
