"""Confusion matrix for the PDRY06 MWL calibration model.

Two evaluations side-by-side:
  Left  — Training-set (deployed model on all calibration data; ~72%)
  Right — 10-fold stratified CV (same SVM pipeline, honest held-out estimate)

Saves:
  results/pdry06/confusion_matrix_pdry06.png
  results/pdry06/confusion_matrix_pdry06.json

Run:
    .venv\Scripts\Activate.ps1
    python scripts/_tmp_confusion_matrix_pdry06.py
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
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

import calibrate_participant as _cal_mod  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402
from ml.dataset import LABEL_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SRATE      = 128.0
CAL_K      = 35
CAL_C      = 1.0
SEED       = 42
CV_SPLITS  = 10

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
FEAT_CFG  = _REPO / "config" / "eeg_feature_extraction.yaml"
OUT_DIR   = _REPO / "results" / "pdry06"
OUT_FIG   = OUT_DIR / "confusion_matrix_pdry06.png"
OUT_JSON  = OUT_DIR / "confusion_matrix_pdry06.json"

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]

# Deployed artefacts
pipeline = joblib.load(MODEL_DIR / "pipeline.pkl")
selector = joblib.load(MODEL_DIR / "selector.pkl")
with open(MODEL_DIR / "norm_stats.json") as f:
    _ns = json.load(f)
NORM_MEAN = np.array(_ns["mean"])
NORM_STD  = np.array(_ns["std"])
NORM_STD[NORM_STD < 1e-12] = 1.0
N_CLASSES = int(_ns.get("n_classes", 3))
print(f"Model: {N_CLASSES}-class\n")

CAL_XDFS = [
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf",
    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf",
]

# ---------------------------------------------------------------------------
# Load calibration features + labels
# ---------------------------------------------------------------------------
region_map = _build_region_map(FEAT_CFG, CH_NAMES)

all_X: list[np.ndarray] = []
all_y: list[int] = []

for xdf_path in CAL_XDFS:
    results = _cal_mod._load_xdf_block(xdf_path, CH_NAMES)
    if results is None:
        sys.exit(f"ERROR: could not load {xdf_path.name}")
    for epochs, level in results:
        X, _ = _extract_feat(epochs, SRATE, region_map)
        X_norm = (X - NORM_MEAN) / NORM_STD
        all_X.append(X_norm)
        all_y.extend([LABEL_MAP[level]] * len(X))
    tag = "C1" if "c1" in xdf_path.stem else "C2"
    n = sum(e.shape[0] for e, _ in results)
    print(f"  {tag}: {n} windows")

X_cal  = np.concatenate(all_X)
y_true = np.array(all_y, dtype=np.int64)
total  = len(y_true)

CLASS_NAMES = ["LOW", "MODERATE", "HIGH"]

# ---------------------------------------------------------------------------
# (A) Training-set evaluation — deployed model
# ---------------------------------------------------------------------------
X_sel    = selector.transform(X_cal)
y_pred_train = pipeline.predict(X_sel)
acc_train = float((y_pred_train == y_true).mean())
cm_train  = confusion_matrix(y_true, y_pred_train, labels=[0, 1, 2])

print(f"\nTraining-set accuracy: {acc_train:.1%}  (n={total})")
print(f"  {'Class':<10}  {'n':>5}  {'correct':>8}  {'acc':>6}")
for lbl, name in zip([0, 1, 2], CLASS_NAMES):
    mask = y_true == lbl
    if mask.any():
        correct = int((y_pred_train[mask] == lbl).sum())
        print(f"  {name:<10}  {mask.sum():>5}  {correct:>8}  {correct/mask.sum():>6.1%}")

# ---------------------------------------------------------------------------
# (B) 10-fold stratified CV
# ---------------------------------------------------------------------------
print(f"\n{CV_SPLITS}-fold stratified CV ...")
cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
y_pred_cv = np.empty(total, dtype=np.int64)

for fold_i, (tr_idx, te_idx) in enumerate(cv.split(X_cal, y_true)):
    Xtr, ytr = X_cal[tr_idx], y_true[tr_idx]
    Xte      = X_cal[te_idx]

    k   = min(CAL_K, Xtr.shape[1])
    sel = SelectKBest(f_classif, k=k)
    Xtr_s = sel.fit_transform(Xtr, ytr)
    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr_s)
    svc   = SVC(kernel="linear", C=CAL_C, class_weight="balanced",
                probability=True, random_state=SEED)
    svc.fit(Xtr_s, ytr)

    Xte_s = sc.transform(sel.transform(Xte))
    y_pred_cv[te_idx] = svc.predict(Xte_s)
    fold_acc = float((svc.predict(Xte_s) == y_true[te_idx]).mean())
    print(f"  fold {fold_i:2d}: acc={fold_acc:.1%}  n_te={len(te_idx)}")

acc_cv = float((y_pred_cv == y_true).mean())
cm_cv  = confusion_matrix(y_true, y_pred_cv, labels=[0, 1, 2])

print(f"\n10-fold CV accuracy: {acc_cv:.1%}  (n={total})")
print(f"  {'Class':<10}  {'n':>5}  {'correct':>8}  {'acc':>6}")
per_class: dict[str, dict] = {}
for lbl, name in zip([0, 1, 2], CLASS_NAMES):
    mask = y_true == lbl
    if mask.any():
        correct = int((y_pred_cv[mask] == lbl).sum())
        print(f"  {name:<10}  {mask.sum():>5}  {correct:>8}  {correct/mask.sum():>6.1%}")
        per_class[name] = {"n": int(mask.sum()), "correct": correct,
                           "recall": round(correct / mask.sum(), 4)}

# ---------------------------------------------------------------------------
# Save JSON summary
# ---------------------------------------------------------------------------
summary = {
    "participant": "PDRY06",
    "n_windows": total,
    "training_set": {
        "accuracy": round(acc_train, 4),
        "confusion_matrix": cm_train.tolist(),
    },
    "cv_10fold": {
        "accuracy": round(acc_cv, 4),
        "confusion_matrix": cm_cv.tolist(),
        "per_class": per_class,
        "n_splits": CV_SPLITS,
        "seed": SEED,
    },
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(summary, indent=2))
print(f"\nJSON saved: {OUT_JSON}")

# ---------------------------------------------------------------------------
# Plot — two confusion matrices side-by-side
# ---------------------------------------------------------------------------
def _annotate_cm(ax, cm: np.ndarray, n_classes: int) -> None:
    """Replace cell text with 'count\n(pct%)' row-normalised."""
    for r in range(n_classes):
        for c in range(n_classes):
            count = cm[r, c]
            pct   = count / cm[r].sum() * 100 if cm[r].sum() > 0 else 0.0
            ax.texts[r * n_classes + c].set_text(f"{count}\n({pct:.0f}%)")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Training-set
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=CLASS_NAMES)
disp_train.plot(ax=axes[0], colorbar=False, cmap="Blues")
_annotate_cm(axes[0], cm_train, 3)
axes[0].set_title(
    f"Training-set (deployed model)\nAccuracy = {acc_train:.1%}  (n={total})",
    fontsize=11,
)
axes[0].set_ylabel("True label", fontsize=10)
axes[0].set_xlabel("Predicted label", fontsize=10)

# 10-fold CV
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=CLASS_NAMES)
disp_cv.plot(ax=axes[1], colorbar=False, cmap="Blues")
_annotate_cm(axes[1], cm_cv, 3)
axes[1].set_title(
    f"10-fold stratified CV\nAccuracy = {acc_cv:.1%}  (n={total})",
    fontsize=11,
)
axes[1].set_ylabel("True label", fontsize=10)
axes[1].set_xlabel("Predicted label", fontsize=10)

fig.suptitle(
    "PDRY06 MWL model — 3-class confusion matrices (calibration data)",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUT_FIG}")
