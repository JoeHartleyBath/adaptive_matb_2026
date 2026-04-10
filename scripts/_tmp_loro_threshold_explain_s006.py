"""Explain why the LORO pooled Youden J threshold is so low (0.033).

The pooled J is 0.228 (good) but the argmax threshold is 0.033 (near-zero).
This script shows WHY by breaking down:

  1. Per-fold p_high distributions by class
     — histograms showing each fold's probability range and class separation
  2. Per-fold ROC curves + pooled ROC, with the argmax threshold marked
  3. Youden J vs threshold curve (pooled) — visualises WHERE and WHY the peak lands
  4. Class-mean p_high table per fold
  5. The "probability scale mismatch" between fold 0 and fold 1

Key hypothesis: fold 1 (train=C1, test=C2) produces much lower absolute p_high
values than fold 0 (train=C2, test=C1), so pooling them shifts the argmax J
towards the left (low threshold end) of the combined probability axis.

Run:
    $env:MATB_SCENARIO_OFFSET_S = "0.943"
    .venv\\Scripts\\python.exe scripts/_tmp_loro_threshold_explain_s006.py
"""
from __future__ import annotations

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
from build_mwl_training_dataset import WINDOW_CONFIG
from eeg.extract_features import _build_region_map, _extract_feat
from ml.dataset import LABEL_MAP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE = 128.0
CAL_K = 35
CAL_C = 1.0
SEED  = 42

PHYSIO   = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")
FEAT_CFG = _REPO / "config" / "eeg_feature_extraction.yaml"
META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

CAL_XDFS = [
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
]
TAGS = ["C1", "C2"]

LOW_LBL  = LABEL_MAP["LOW"]
HIGH_LBL = LABEL_MAP["HIGH"]
MOD_LBL  = LABEL_MAP["MODERATE"]
LABEL_NAMES = {LOW_LBL: "LOW", MOD_LBL: "MOD", HIGH_LBL: "HIGH"}

# ---------------------------------------------------------------------------
# Load raw features
# ---------------------------------------------------------------------------
print("Loading XDFs (per-run LOW norm) ...\n")
region_map = _build_region_map(FEAT_CFG, CH_NAMES)
feat_names: list[str] = []
xdf_raw_X, xdf_y, xdf_low_X = [], [], []

for xdf_path, tag in zip(CAL_XDFS, TAGS):
    print(f"  {tag}: ", end="", flush=True)
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        sys.exit(1)
    all_feats, all_labels, low_feats = [], [], []
    for epochs, level_str in results:
        lbl = LABEL_MAP[level_str]
        X_raw, names = _extract_feat(epochs, SRATE, region_map)
        if not feat_names:
            feat_names = names
        all_feats.append(X_raw)
        all_labels.extend([lbl] * len(X_raw))
        if lbl == LOW_LBL:
            low_feats.append(X_raw)
    xdf_raw_X.append(np.concatenate(all_feats))
    xdf_y.append(np.array(all_labels, dtype=np.int64))
    xdf_low_X.append(np.concatenate(low_feats) if low_feats else np.concatenate(all_feats))

# Per-run LOW norms
def _norm_stats(X):
    m = X.mean(0); s = X.std(0); s[s < 1e-12] = 1.0; return m, s

norms_C = [_norm_stats(Xl) for Xl in xdf_low_X]
xdf_C   = [(X - mn) / st for X, (mn, st) in zip(xdf_raw_X, norms_C)]

# ---------------------------------------------------------------------------
# Run LORO folds and capture per-fold p_high arrays
# ---------------------------------------------------------------------------
fold_data: list[dict] = []   # one dict per fold

for fold_i in range(2):
    train_X = np.concatenate([xdf_C[j] for j in range(2) if j != fold_i])
    train_y = np.concatenate([xdf_y[j] for j in range(2) if j != fold_i])
    test_X  = xdf_C[fold_i]
    test_y  = xdf_y[fold_i]

    sel = SelectKBest(f_classif, k=min(CAL_K, train_X.shape[1]))
    Xtr = sel.fit_transform(train_X, train_y)
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    svc = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
              probability=True, random_state=SEED)
    svc.fit(Xtr, train_y)

    Xte    = sc.transform(sel.transform(test_X))
    p_high = svc.predict_proba(Xte)[:, -1]
    y_bin  = (test_y == HIGH_LBL).astype(int)

    fpr, tpr, thr_arr = roc_curve(y_bin, p_high, drop_intermediate=False)
    j_arr = tpr - fpr
    best_i = int(np.argmax(j_arr))

    fold_data.append({
        "fold_i":     fold_i,
        "test_tag":   TAGS[fold_i],
        "train_tag":  TAGS[1 - fold_i],
        "p_high":     p_high,
        "test_y":     test_y,
        "y_bin":      y_bin,
        "fpr":        fpr,
        "tpr":        tpr,
        "thr_arr":    thr_arr,
        "j_arr":      j_arr,
        "best_i":     best_i,
        "fold_j":     float(j_arr[best_i]),
        "fold_thr":   float(thr_arr[best_i]),
        "auc":        float(roc_auc_score(y_bin, p_high)),
    })
    print(f"  fold {fold_i} (train={TAGS[1-fold_i]}, test={TAGS[fold_i]}):  "
          f"J={j_arr[best_i]:.3f}  AUC={roc_auc_score(y_bin, p_high):.3f}  "
          f"fold_thr={thr_arr[best_i]:.4f}")

# Pooled
all_p = np.concatenate([d["p_high"] for d in fold_data])
all_y = np.concatenate([d["y_bin"]  for d in fold_data])
fpr_pool, tpr_pool, thr_pool = roc_curve(all_y, all_p, drop_intermediate=False)
j_pool  = tpr_pool - fpr_pool
best_pool = int(np.argmax(j_pool))
pooled_thr = float(thr_pool[best_pool])
pooled_j   = float(j_pool[best_pool])
print(f"\n  Pooled:  J={pooled_j:.3f}  threshold={pooled_thr:.4f}")

# ---------------------------------------------------------------------------
# Print class-mean p_high table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CLASS-MEAN p_high BY FOLD")
print("=" * 70)
print(f"  {'Fold':<30}  {'LOW_mean':>9}  {'MOD_mean':>9}  {'HIGH_mean':>9}  "
      f"  {'range_LOW':>10}  {'range_HIGH':>11}")
print("  " + "-" * 75)
for d in fold_data:
    ph = d["p_high"]; ty = d["test_y"]
    for lbl, lname in LABEL_NAMES.items():
        mask = ty == lbl
        if mask.any():
            print(f"      {lname}: {ph[mask].mean():.3f}  (std={ph[mask].std():.3f}, "
                  f"min={ph[mask].min():.3f}, max={ph[mask].max():.3f})", end="")
    print()
    lo = ph[ty == LOW_LBL]
    hi = ph[ty == HIGH_LBL]
    print(f"  fold {d['fold_i']} train={d['train_tag']}, test={d['test_tag']}")
    for lbl, lname in LABEL_NAMES.items():
        mask = ty == lbl
        if mask.any():
            p = ph[mask]
            print(f"    {lname:<10}  mean={p.mean():6.3f}  std={p.std():5.3f}  "
                  f"median={np.median(p):6.3f}  p10={np.percentile(p,10):5.3f}  "
                  f"p90={np.percentile(p,90):5.3f}")

print(f"\n  NOTE: pooled argmax J threshold = {pooled_thr:.4f}")
print(f"  This is chosen to maximise (TPR - FPR) across BOTH folds' p_high values combined.")
print(f"  If fold 1 HIGH-class p_high values are << fold 0 HIGH-class p_high values,")
print(f"  the pooled threshold shifts left (toward 0) to capture fold 1's HIGH windows.")

# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLD_COLOURS = ["steelblue", "darkorange"]
CLASS_COLOURS = {LOW_LBL: "royalblue", MOD_LBL: "goldenrod", HIGH_LBL: "firebrick"}

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "S006 LORO Youden J diagnostic — per-run LOW norm\n"
    "Why does pooled J=0.228 but threshold=0.033?",
    fontsize=12, fontweight="bold",
)

# ---- Layout: 3 rows, each with 2 columns ----
# Row 1: p_high histograms per fold (col 0 = fold 0, col 1 = fold 1)
# Row 2: ROC per fold + pooled ROC (shared)
# Row 3: J-score vs threshold (per fold + pooled)

gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.30, left=0.07, right=0.97,
                       top=0.91, bottom=0.05)

# ---- Row 1: Histograms ----
for col, d in enumerate(fold_data):
    ax = fig.add_subplot(gs[0, col])
    ph = d["p_high"]; ty = d["test_y"]
    bins = np.linspace(0, 1, 41)
    for lbl, lname in LABEL_NAMES.items():
        mask = ty == lbl
        if not mask.any():
            continue
        ax.hist(ph[mask], bins=bins, alpha=0.55,
                color=CLASS_COLOURS[lbl], label=f"{lname} (n={mask.sum()})",
                density=True)
    ax.axvline(d["fold_thr"], color="forestgreen", linestyle="-.", linewidth=1.2,
               label=f"fold opt thr={d['fold_thr']:.3f}")
    ax.axvline(pooled_thr, color="crimson", linestyle="--", linewidth=1.2,
               label=f"pooled thr={pooled_thr:.3f}")
    ax.set_xlabel("p_high", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(
        f"Fold {d['fold_i']}: train={d['train_tag']}, test={d['test_tag']}\n"
        f"J={d['fold_j']:.3f}  AUC={d['auc']:.3f}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    # Annotate class means
    for lbl, lname in LABEL_NAMES.items():
        mask = ty == lbl
        if mask.any():
            ax.axvline(ph[mask].mean(), color=CLASS_COLOURS[lbl],
                       linestyle=":", linewidth=0.8, alpha=0.8)

# ---- Row 2: ROC curves ----
ax_roc = fig.add_subplot(gs[1, :])   # span both columns

for d in fold_data:
    col = FOLD_COLOURS[d["fold_i"]]
    ax_roc.plot(d["fpr"], d["tpr"], color=col, linewidth=1.5,
                label=(f"Fold {d['fold_i']}: train={d['train_tag']}, test={d['test_tag']}  "
                       f"AUC={d['auc']:.3f}  J={d['fold_j']:.3f}  "
                       f"fold_thr={d['fold_thr']:.3f}"))
    # Mark the per-fold optimal point
    bi = d["best_i"]
    ax_roc.scatter(d["fpr"][bi], d["tpr"][bi], s=60, zorder=5,
                   color=col, marker="o", edgecolors="k", linewidths=0.6)

ax_roc.plot(fpr_pool, tpr_pool, color="black", linewidth=2.0, linestyle="-",
            label=f"Pooled  AUC={roc_auc_score(all_y, all_p):.3f}  J={pooled_j:.3f}  thr={pooled_thr:.3f}")
ax_roc.scatter(fpr_pool[best_pool], tpr_pool[best_pool], s=80, zorder=6,
               color="crimson", marker="*", edgecolors="k", linewidths=0.6,
               label=f"Pooled argmax(J) at thr={pooled_thr:.3f}")

ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.6, alpha=0.4)
ax_roc.set_xlabel("FPR (1 - Specificity)", fontsize=9)
ax_roc.set_ylabel("TPR (Sensitivity)", fontsize=9)
ax_roc.set_title("ROC curves — per fold and pooled", fontsize=10)
ax_roc.legend(fontsize=7.5, loc="lower right")
ax_roc.tick_params(labelsize=8)
ax_roc.set_xlim(-0.02, 1.02)
ax_roc.set_ylim(-0.02, 1.02)

# ---- Row 3: J-score vs threshold ----
ax_j0 = fig.add_subplot(gs[2, 0])
ax_j1 = fig.add_subplot(gs[2, 1])

for col_ax, d in zip([ax_j0, ax_j1], fold_data):
    thr_a = d["thr_arr"][:-1]      # roc_curve appends an extra sentinel
    j_a   = d["j_arr"][:-1]
    col   = FOLD_COLOURS[d["fold_i"]]
    col_ax.plot(thr_a, j_a, color=col, linewidth=1.5,
                label=f"fold {d['fold_i']} J(thr)")
    # Pooled J curve on same axis
    thr_pl = thr_pool[:-1]; j_pl = j_pool[:-1]
    col_ax.plot(thr_pl, j_pl, color="black", linewidth=1.0, linestyle="--",
                alpha=0.6, label="pooled J(thr)")
    col_ax.axvline(d["fold_thr"], color="forestgreen", linestyle="-.", linewidth=1.0,
                   label=f"fold opt={d['fold_thr']:.3f}")
    col_ax.axvline(pooled_thr, color="crimson", linestyle="--", linewidth=1.0,
                   label=f"pooled opt={pooled_thr:.3f}")
    col_ax.axhline(0.10, color="grey", linestyle=":", linewidth=0.8, alpha=0.6,
                   label="min J=0.10")
    col_ax.set_xlabel("Threshold", fontsize=8)
    col_ax.set_ylabel("Youden J = TPR - FPR", fontsize=8)
    col_ax.set_title(
        f"Fold {d['fold_i']}: J vs threshold\n"
        f"(fold argmax={d['fold_thr']:.3f}, pooled argmax={pooled_thr:.3f})",
        fontsize=9,
    )
    col_ax.legend(fontsize=7, loc="upper right")
    col_ax.tick_params(labelsize=7)
    col_ax.set_xlim(-0.02, 1.02)
    col_ax.set_ylim(-0.05, 0.60)

out_fig = _REPO / "results" / "figures" / "s006_loro_threshold_explanation.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_fig, dpi=150)
plt.close(fig)
print(f"\nFigure saved: {out_fig}")

# ---------------------------------------------------------------------------
# Summary explanation
# ---------------------------------------------------------------------------
d0, d1 = fold_data
ph_low_0  = d0["p_high"][d0["test_y"] == LOW_LBL].mean()
ph_high_0 = d0["p_high"][d0["test_y"] == HIGH_LBL].mean()
ph_low_1  = d1["p_high"][d1["test_y"] == LOW_LBL].mean()
ph_high_1 = d1["p_high"][d1["test_y"] == HIGH_LBL].mean()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Fold 0 (train=C2 test=C1):  mean p_high | LOW={ph_low_0:.3f}  HIGH={ph_high_0:.3f}")
print(f"  Fold 1 (train=C1 test=C2):  mean p_high | LOW={ph_low_1:.3f}  HIGH={ph_high_1:.3f}")
print()
scale_diff = ph_high_0 - ph_high_1
print(f"  HIGH-class mean p_high difference across folds: {scale_diff:+.3f}")
if scale_diff > 0.15:
    print(f"  => Fold 0 HIGH windows score much higher than fold 1 HIGH windows.")
    print(f"     When pooled, the argmax J threshold must be low enough to capture")
    print(f"     fold 1's HIGH windows (which sit near {ph_high_1:.3f}).")
    print(f"     This gives pooled_thr={pooled_thr:.4f}.")
    print()
    print(f"  The two folds have DIFFERENT probability scales:")
    print(f"    - Fold 0 opt threshold: {d0['fold_thr']:.4f}  (operating range ~{ph_low_0:.2f}..{ph_high_0:.2f})")
    print(f"    - Fold 1 opt threshold: {d1['fold_thr']:.4f}  (operating range ~{ph_low_1:.2f}..{ph_high_1:.2f})")
    print(f"  Pooling mixes these two ranges and finds the single threshold")
    print(f"  that maximises J across both — which is near the lower range end.")
print()
print(f"  Implication for deployment:")
print(f"    The per-run LOW norm removes the *relative* amplitude offset between")
print(f"    C1 and C2, allowing both folds to discriminate. But each fold's absolute")
print(f"    p_high scale is different because each is a different model (C1-trained vs C2-trained).")
print(f"    The training-set threshold ({0.214:.3f}) comes from a model trained on ALL data")
print(f"    and is a better guide to the deployment operating point.")
print()
print("--- End ---")
