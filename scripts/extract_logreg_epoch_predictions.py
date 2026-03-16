"""Extract per-epoch LogReg predictions and model coefficients.

Runs the frozen LOSO loop with the WS-weak (C=0.1, 180s) personalisation
strategy and saves:
  1. Per-epoch predictions CSV  — one row per epoch per participant
  2. Per-fold model coefficients JSON — selected features, coef_, intercept_

These outputs feed the beyond-AUC validation analyses (temporal dynamics,
neurophysiological plausibility, convergent validity).

Usage:
    python scripts/extract_logreg_epoch_predictions.py
    python scripts/extract_logreg_epoch_predictions.py --only P05 P07
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import (  # noqa: E402
    calibration_norm_features,
    load_baseline_from_cache,
)

_NORM_CACHE  = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_QC_CONFIG   = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_OUT_DIR     = _REPO_ROOT / "results" / "test_pretrain"

_DEFAULT_CSV  = _OUT_DIR / "logreg_epoch_predictions.csv"
_DEFAULT_JSON = _OUT_DIR / "logreg_fold_coefficients.json"

# Frozen config (dc_logreg_hyperparameter_plateau + dc_logreg_personalisation)
_LOGREG_K       = 30
_LOGREG_C       = 0.001
_WARM_C         = 0.1       # WS-weak
_CAL_DURATION   = 60        # seconds per label
_STEP_S         = 0.5       # epoch step in the HDF5 export
_GAP_RADIUS     = 3         # gap zone around cal chunks (in epochs)

SEED = 42


def _load_exclude(cfg_path: Path) -> set[str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_EXCLUDE = _load_exclude(_QC_CONFIG)


# ===================================================================
# Data loading (extends logreg_personalisation_comparison.load_data
# to also return task_bidx)
# ===================================================================

def load_data() -> tuple[
    dict[str, np.ndarray],   # X_by  (calibration-normalised features)
    dict[str, np.ndarray],   # y_by  (binary labels)
    dict[str, np.ndarray],   # bidx_by (block temporal index 0-3)
    list[str],               # feat_names
    list[str],               # pids
]:
    if not _NORM_CACHE.exists():
        raise SystemExit(
            f"ERROR: Norm cache not found at {_NORM_CACHE}.\n"
            "       Run causal_norm_comparison.py first to build it."
        )

    npz = np.load(_NORM_CACHE, allow_pickle=False)
    npz_pids = list(npz["pids"])
    available = [p for p in npz_pids if p not in _EXCLUDE]
    if not available:
        raise SystemExit("No participants remain after exclusion.")

    feat_names = list(npz["feat_names"])
    X_by_raw: dict[str, np.ndarray] = {}
    y_by: dict[str, np.ndarray] = {}
    bidx_by: dict[str, np.ndarray] = {}
    for pid in available:
        X_by_raw[pid] = npz[f"{pid}_task_X"]
        y_by[pid]     = npz[f"{pid}_task_y"]
        bidx_by[pid]  = npz[f"{pid}_task_bidx"]

    baseline_by = load_baseline_from_cache(_NORM_CACHE, available)
    if baseline_by is None:
        raise SystemExit("Baseline data missing from norm cache.")

    X_by: dict[str, np.ndarray] = {}
    for pid in available:
        bl = baseline_by[pid]
        X_by[pid] = calibration_norm_features(
            X_by_raw[pid], bl["fix_X"], bl["forest_X"], bl["forest_bidx"],
        )

    pids = sorted(available)
    return X_by, y_by, bidx_by, feat_names, pids


# ===================================================================
# Block detection & calibration split (from personalisation_comparison)
# ===================================================================

def _detect_blocks(y: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    n = len(y)
    i = 0
    while i < n:
        label = int(y[i])
        j = i + 1
        while j < n and y[j] == label:
            j += 1
        blocks.append({"label": label, "start": i, "end": j})
        i = j
    return blocks


def _cal_split_indices(
    y: np.ndarray,
    cal_seconds: float,
    rng: np.random.Generator,
) -> tuple[list[int], set[int], list[int]]:
    """Return (cal_indices, gap_indices, test_indices)."""
    n_cal_per_label = int(cal_seconds / _STEP_S)
    blocks = _detect_blocks(y)

    blocks_by_label: dict[int, list[dict]] = {}
    for b in blocks:
        blocks_by_label.setdefault(b["label"], []).append(b)

    cal_indices: list[int] = []
    gap_indices: set[int]  = set()

    for label in sorted(blocks_by_label.keys()):
        label_blocks = blocks_by_label[label]
        n_blocks = len(label_blocks)
        base_per_block = n_cal_per_label // n_blocks
        remainder      = n_cal_per_label % n_blocks

        for bi, block in enumerate(label_blocks):
            block_start = block["start"]
            block_size  = block["end"] - block["start"]
            n_from_block = base_per_block + (1 if bi < remainder else 0)
            n_from_block = min(n_from_block, block_size)
            if n_from_block <= 0:
                continue

            max_start = block_size - n_from_block
            offset = 0 if max_start <= 0 else int(rng.integers(0, max_start + 1))

            chunk_start = block_start + offset
            chunk_end   = chunk_start + n_from_block
            cal_indices.extend(range(chunk_start, chunk_end))

            for g in range(1, _GAP_RADIUS + 1):
                before = chunk_start - g
                after  = chunk_end - 1 + g
                if before >= block_start:
                    gap_indices.add(before)
                if after < block["end"]:
                    gap_indices.add(after)

    cal_set     = set(cal_indices)
    gap_only    = gap_indices - cal_set
    test_indices = sorted(set(range(len(y))) - cal_set - gap_only)
    return cal_indices, gap_only, test_indices


# ===================================================================
# Model helpers
# ===================================================================

def _make_logreg(C: float = _LOGREG_C) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000,
                                   class_weight="balanced",
                                   random_state=SEED)),
    ])


def _train_group_model(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    held_out_pid: str,
) -> tuple[SelectKBest, Pipeline]:
    train_pids = sorted(p for p in X_by if p != held_out_pid)
    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])

    selector = SelectKBest(f_classif, k=_LOGREG_K)
    X_train_sel = selector.fit_transform(X_train, y_train)

    pipe = _make_logreg(C=_LOGREG_C)
    pipe.fit(X_train_sel, y_train)
    return selector, pipe


def _personalise_ws_weak(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> LogisticRegression:
    """Warm-start from group weights with weak L2 (C=0.1)."""
    sc        = group_pipe.named_steps["sc"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel = selector.transform(X_cal)
    X_cal_sc  = sc.transform(X_cal_sel)

    clf = LogisticRegression(C=_WARM_C, max_iter=2000,
                             warm_start=True, class_weight="balanced",
                             random_state=SEED)
    clf.classes_   = group_clf.classes_.copy()
    clf.coef_      = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(X_cal_sc, y_cal)
    return clf


def _youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


# ===================================================================
# Main extraction loop
# ===================================================================

def run_extraction(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    bidx_by: dict[str, np.ndarray],
    feat_names: list[str],
    pids: list[str],
) -> tuple[list[dict], dict]:
    """Run LOSO, extract per-epoch predictions and per-fold coefficients.

    Returns (epoch_rows, fold_coefs).
    """
    epoch_rows: list[dict] = []
    fold_coefs: dict = {}
    n_total = len(pids)

    for i, pid in enumerate(pids):
        t0 = time.time()
        print(f"  [{i+1:>2}/{n_total}] {pid} ...", end=" ", flush=True)

        # 1. Train group model
        selector, group_pipe = _train_group_model(X_by, y_by, pid)
        sc = group_pipe.named_steps["sc"]

        # 2. Cal / test split (same RNG as personalisation_comparison)
        rng = np.random.default_rng(SEED + hash(pid) % (2**31))
        cal_idx, gap_idx, test_idx = _cal_split_indices(
            y_by[pid], _CAL_DURATION, rng)

        X_pid = X_by[pid]
        y_pid = y_by[pid]
        bidx  = bidx_by[pid]

        X_cal = X_pid[cal_idx]
        y_cal = y_pid[cal_idx]

        if len(X_cal) < 4 or len(np.unique(y_cal)) < 2:
            print("SKIP (insufficient cal data)")
            continue

        # 3. Train personalised model
        personal_clf = _personalise_ws_weak(selector, group_pipe, X_cal, y_cal)

        # 4. Predict ALL epochs with both models
        X_all_sel = selector.transform(X_pid)
        X_all_sc  = sc.transform(X_all_sel)

        probs_group    = group_pipe.predict_proba(X_all_sel)[:, 1]
        probs_personal = personal_clf.predict_proba(X_all_sc)[:, 1]

        # 5. Youden threshold from cal data
        probs_cal_personal = personal_clf.predict_proba(
            sc.transform(selector.transform(X_cal)))[:, 1]
        threshold = _youden_threshold(y_cal, probs_cal_personal)

        # 6. Assign roles
        cal_set = set(cal_idx)
        gap_set = set(gap_idx)
        roles = []
        for idx in range(len(y_pid)):
            if idx in cal_set:
                roles.append("cal")
            elif idx in gap_set:
                roles.append("gap")
            else:
                roles.append("test")

        # 7. Compute epoch_within_block
        block_counters: dict[int, int] = {}
        for idx in range(len(y_pid)):
            bi = int(bidx[idx])
            ewb = block_counters.get(bi, 0)
            block_counters[bi] = ewb + 1

            epoch_rows.append({
                "pid":              pid,
                "block_idx":        bi,
                "epoch_within_block": ewb,
                "epoch_idx":        idx,
                "y_true":           int(y_pid[idx]),
                "y_prob_group":     round(float(probs_group[idx]), 6),
                "y_prob_personal":  round(float(probs_personal[idx]), 6),
                "role":             roles[idx],
            })

        # 8. Coefficients
        selected_mask = selector.get_support()
        selected_names = [feat_names[j] for j in range(len(feat_names))
                          if selected_mask[j]]
        group_clf = group_pipe.named_steps["clf"]

        # Test-set metrics for reference
        y_test = y_pid[test_idx]
        p_test = probs_personal[test_idx]
        auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan")
        preds_t = (p_test >= threshold).astype(int)
        bal_acc = float(balanced_accuracy_score(y_test, preds_t))

        fold_coefs[pid] = {
            "selected_features": selected_names,
            "group_coef":        group_clf.coef_[0].tolist(),
            "group_intercept":   float(group_clf.intercept_[0]),
            "personal_coef":     personal_clf.coef_[0].tolist(),
            "personal_intercept": float(personal_clf.intercept_[0]),
            "threshold":         threshold,
            "test_auc":          auc,
            "test_bal_acc":      bal_acc,
            "n_cal":             len(cal_idx),
            "n_gap":             len(gap_idx),
            "n_test":            len(test_idx),
        }

        elapsed = time.time() - t0
        print(f"AUC={auc:.3f}  bal_acc={bal_acc:.3f}  [{elapsed:.1f}s]")

    return epoch_rows, fold_coefs


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Extract per-epoch LogReg predictions for beyond-AUC validation")
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=_DEFAULT_JSON)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Run only these participant IDs")
    args = parser.parse_args()

    print("LogReg Epoch Prediction Extractor")
    print(f"  Strategy       : WS-weak (C={_WARM_C})")
    print(f"  Cal duration   : {_CAL_DURATION}s per label")
    print(f"  Gap radius     : +-{_GAP_RADIUS} epochs")
    print(f"  Seed           : {SEED}")
    print()

    X_by, y_by, bidx_by, feat_names, pids = load_data()
    print(f"  Participants   : {len(pids)}")
    print(f"  Features       : {len(feat_names)}")
    print()

    if args.only:
        pids = [p for p in pids if p in args.only]
        if not pids:
            print(f"ERROR: None of {args.only} found in included participants.")
            sys.exit(1)
        print(f"  Running for    : {pids}")
        print()

    t_start = time.time()
    epoch_rows, fold_coefs = run_extraction(
        X_by, y_by, bidx_by, feat_names, pids)
    t_total = (time.time() - t_start) / 60

    # Save CSV
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    header = "pid,block_idx,epoch_within_block,epoch_idx,y_true,y_prob_group,y_prob_personal,role"
    with open(args.csv, "w", newline="") as f:
        f.write(header + "\n")
        for row in epoch_rows:
            f.write(
                f"{row['pid']},{row['block_idx']},{row['epoch_within_block']},"
                f"{row['epoch_idx']},{row['y_true']},{row['y_prob_group']},"
                f"{row['y_prob_personal']},{row['role']}\n"
            )
    print(f"  Saved CSV  : {args.csv}  ({len(epoch_rows)} rows)")

    # Save JSON
    out_data = {
        "config": {
            "strategy": "warm_start_weak_l2",
            "warm_C": _WARM_C,
            "group_C": _LOGREG_C,
            "K": _LOGREG_K,
            "cal_duration_s": _CAL_DURATION,
            "gap_radius": _GAP_RADIUS,
            "step_s": _STEP_S,
            "n_participants": len(pids),
            "n_features": len(feat_names),
            "feat_names": feat_names,
            "seed": SEED,
        },
        "folds": fold_coefs,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out_data, indent=2))
    print(f"  Saved JSON : {args.json}  ({len(fold_coefs)} folds)")
    print(f"  Wall time  : {t_total:.1f} min")


if __name__ == "__main__":
    main()
