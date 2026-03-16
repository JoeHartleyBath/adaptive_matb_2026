"""Personalisation vs from-scratch comparison for MWL classification.

For each of the 28 included participants (LOSO), this script:
  1. Trains a group RBF model on the other 27 participants.
  2. Splits the held-out participant's data into calibration and test sets
     by randomly sampling from all 4 VR-TSST condition blocks (both stress
     levels for each workload level), with a gap buffer around each sampled
     epoch to prevent window-overlap leakage.
  3. Evaluates six strategies on the test portion:
       A  group_only     — group model applied directly, cal data ignored
       B  finetune_head  — refit LogReg head on Nystroem-projected cal data
       C  prior_reg      — like B, but warm-start from group weights + strong L2
       D  scratch_rbf    — full RBF pipeline trained on cal data only
       E  scratch_logreg — plain LogReg trained on cal data only
       F  scratch_rf     — Random Forest trained on cal data only

Calibration durations (per label): 30 s, 60 s, 90 s, 120 s, 180 s.
Epochs are sampled randomly (seeded) from each of the 4 condition blocks,
distributed equally across both blocks of the same label.  A gap of ±4 epochs
around each sampled epoch is excluded from the test set to prevent leakage
from the 2 s window / 0.5 s step overlap.

Reuses the 54-feature cache from personalised_logreg.py (same _EXCLUDE,
_FIXED, feat_v="v2_1f_wpli") and per-participant z-scoring.

Usage:
    # Full run (28 participants × 5 durations × 6 strategies)
    python scripts/personalisation_comparison.py

    # Quick dry-run on one participant
    python scripts/personalisation_comparison.py --only P05

    # Override output path
    python scripts/personalisation_comparison.py --out results/test_pretrain/custom.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import load_baseline_from_cache, prepare_mixed_norm  # noqa: E402

_DATASET       = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_FEATURE_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "feature_cache.npz"
_NORM_CACHE    = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_DEFAULT_OUT   = _REPO_ROOT / "results" / "test_pretrain" / "personalisation_comparison.json"

# Must match personalised_logreg.py exactly
_EXCLUDE = {
    "P16", "P21", "P27", "P34", "P37", "P43", "P45", "P31", "P13",
    "P04", "P06", "P09", "P20", "P23", "P33", "P35", "P39", "P44", "P46",
}

_FIXED = {
    "Delta":  (1.0,  4.0),
    "Theta":  (4.0,  7.5),
    "Alpha":  (7.5, 12.0),
    "Beta":  (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}

# RBF frozen config (from DC-08 ablation)
_RBF_GAMMA  = 0.01
_RBF_C      = 1.0
_RBF_NYS    = 300
_RBF_K      = 30           # SelectKBest k
_N_INNER    = 5             # inner CV folds for group model training

# Calibration durations (seconds per label)
_CAL_DURATIONS = [30, 60, 90, 120, 180]
# Window step used in the HDF5 export (determines epochs-per-second)
_STEP_S     = 0.5
# Gap radius: epochs within this distance of a cal chunk boundary (within
# the same block) are excluded from test.  At step=0.5s / window=2.0s,
# epochs at distance 4 have 0% overlap, so gap=3 prevents any data sharing.
_GAP_RADIUS = 3

SEED = 42


# ===================================================================
# Cache loading (must match personalised_logreg.py cache format)
# ===================================================================

def _cache_key(dataset_path: Path) -> str:
    """SHA-256 cache key — must match personalised_logreg.py exactly."""
    manifest = dataset_path / "manifest.json"
    mtime  = str(manifest.stat().st_mtime) if manifest.exists() else "missing"
    excl   = str(sorted(_EXCLUDE))
    bands  = str(sorted((k, v) for k, v in _FIXED.items()))
    feat_v = "v3_continuous"
    raw    = "|".join([mtime, excl, bands, feat_v])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_feature_cache(
    cache_path: Path, key: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]] | None:
    """Load cached features if the key matches."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if str(data["cache_key"]) != key:
            return None
        pids       = list(data["pids"])
        feat_names = list(data["feat_names"])
        X_by = {pid: data[f"X_{pid}"] for pid in pids}
        y_by = {pid: data[f"y_{pid}"] for pid in pids}
        return X_by, y_by, feat_names
    except Exception:
        return None


# ===================================================================
# Pipeline factories
# ===================================================================

def _make_rbf(gamma: float, C: float, n_components: int,
              seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("nys", Nystroem(kernel="rbf", gamma=gamma,
                         n_components=n_components, random_state=seed)),
        ("clf", LogisticRegression(C=C, max_iter=1000,
                                   class_weight="balanced",
                                   random_state=seed)),
    ])


def _make_logreg(C: float = 0.001, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000,
                                   class_weight="balanced",
                                   random_state=seed)),
    ])


def _make_rf(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=seed,
            class_weight="balanced")),
    ])


# ===================================================================
# AUC helper
# ===================================================================

def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find the optimal classification threshold via Youden's J statistic.

    Returns (threshold, youden_j).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


# ===================================================================
# Block detection & random calibration split
# ===================================================================

def _detect_blocks(y: np.ndarray) -> list[dict]:
    """Detect contiguous condition blocks from the label sequence.

    The HDF5 stores epochs in chronological order of condition blocks.
    Each participant has 4 blocks (~297 epochs each), 2 per label.
    Returns a list of dicts with keys: label, start, end (exclusive).
    """
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


def _random_cal_split(
    X: np.ndarray,
    y: np.ndarray,
    cal_seconds: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split one participant's data into cal and test sets.

    For each condition block, selects a contiguous chunk of the required size
    starting at a random offset.  A gap of _GAP_RADIUS epochs on each side
    of the chunk (within the same block) is excluded from the test set to
    prevent window-overlap leakage.

    Returns:
        X_cal, y_cal, X_test, y_test, info_dict
    """
    n_cal_per_label = int(cal_seconds / _STEP_S)
    blocks = _detect_blocks(y)

    # Group blocks by label
    blocks_by_label: dict[int, list[dict]] = {}
    for b in blocks:
        blocks_by_label.setdefault(b["label"], []).append(b)

    cal_indices:      list[int] = []
    excluded_indices: set[int]  = set()   # gap zone — neither cal nor test

    for label in sorted(blocks_by_label.keys()):
        label_blocks = blocks_by_label[label]
        n_blocks = len(label_blocks)
        # Distribute cal epochs equally across blocks of this label
        base_per_block = n_cal_per_label // n_blocks
        remainder      = n_cal_per_label % n_blocks

        for bi, block in enumerate(label_blocks):
            block_start = block["start"]
            block_size  = block["end"] - block["start"]
            # This block gets base + 1 if bi < remainder
            n_from_block = base_per_block + (1 if bi < remainder else 0)
            n_from_block = min(n_from_block, block_size)

            if n_from_block <= 0:
                continue

            # Pick a random start position for a contiguous chunk
            max_start = block_size - n_from_block
            if max_start <= 0:
                offset = 0
            else:
                offset = int(rng.integers(0, max_start + 1))

            chunk_start = block_start + offset
            chunk_end   = chunk_start + n_from_block  # exclusive

            cal_indices.extend(range(chunk_start, chunk_end))

            # Mark gap zones at chunk boundaries (within this block only)
            for g in range(1, _GAP_RADIUS + 1):
                before = chunk_start - g
                after  = chunk_end - 1 + g
                if before >= block_start:
                    excluded_indices.add(before)
                if after < block["end"]:
                    excluded_indices.add(after)

    # Test = everything not in cal and not in gap exclusion zone
    cal_set      = set(cal_indices)
    all_indices  = set(range(len(y)))
    test_indices = sorted(all_indices - cal_set - excluded_indices)
    n_gap_only   = len(excluded_indices - cal_set)

    info = {
        "n_cal":      len(cal_indices),
        "n_test":     len(test_indices),
        "n_excluded": n_gap_only,
        "n_blocks":   len(blocks),
        "blocks":     [(b["label"], b["end"] - b["start"]) for b in blocks],
    }
    return (
        X[cal_indices], y[cal_indices],
        X[test_indices], y[test_indices],
        info,
    )


# ===================================================================
# Group model training (LOSO — train on 27, produce fitted pipeline)
# ===================================================================

def _train_group_model(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    held_out_pid: str,
) -> tuple[SelectKBest, Pipeline, dict]:
    """Train the frozen-config RBF pipeline on all participants except held_out_pid.

    Uses inner CV (StratifiedGroupKFold) for K selection and gamma/C tuning,
    then refits on the full training set.

    Returns:
        selector: fitted SelectKBest
        pipe:     fitted RBF Pipeline (StandardScaler -> Nystroem -> LogReg)
        info:     dict with best_k, best_gamma, best_C
    """
    train_pids = sorted(p for p in X_by if p != held_out_pid)
    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])
    groups  = np.concatenate([
        np.full(len(y_by[p]), j, dtype=np.int32)
        for j, p in enumerate(train_pids)
    ])

    # K selection via inner CV with LogReg probe
    k_candidates = [15, 20, 25, 30, 45]
    best_k, best_auc_k = _RBF_K, -1.0
    cv = StratifiedGroupKFold(n_splits=_N_INNER)
    for k in k_candidates:
        fold_aucs = []
        for tr, te in cv.split(X_train, y_train, groups):
            sel  = SelectKBest(f_classif, k=k)
            Xf_tr = sel.fit_transform(X_train[tr], y_train[tr])
            Xf_te = sel.transform(X_train[te])
            probe = _make_logreg(C=0.001, seed=SEED)
            probe.fit(Xf_tr, y_train[tr])
            probs = probe.predict_proba(Xf_te)[:, 1]
            fold_aucs.append(_auc(y_train[te], probs))
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_auc_k:
            best_k, best_auc_k = k, mean_auc

    # Gamma/C grid search (same 6-config grid as rbf_ablation baseline)
    gamma_C_grid = [
        {"gamma": g, "C": c}
        for g in [0.01, 0.05, 0.1]
        for c in [0.1, 1.0]
    ]
    best_params = {"gamma": _RBF_GAMMA, "C": _RBF_C}
    best_auc_gc = -1.0
    for params in gamma_C_grid:
        fold_aucs = []
        for tr, te in cv.split(X_train, y_train, groups):
            sel  = SelectKBest(f_classif, k=best_k)
            Xf_tr = sel.fit_transform(X_train[tr], y_train[tr])
            Xf_te = sel.transform(X_train[te])
            pipe  = _make_rbf(params["gamma"], params["C"], _RBF_NYS, SEED)
            pipe.fit(Xf_tr, y_train[tr])
            probs = pipe.predict_proba(Xf_te)[:, 1]
            fold_aucs.append(_auc(y_train[te], probs))
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_auc_gc:
            best_params, best_auc_gc = params, mean_auc

    # Final refit on full training data
    selector = SelectKBest(f_classif, k=best_k)
    X_train_sel = selector.fit_transform(X_train, y_train)

    pipe = _make_rbf(best_params["gamma"], best_params["C"], _RBF_NYS, SEED)
    pipe.fit(X_train_sel, y_train)

    info = {
        "best_k": best_k,
        "best_gamma": best_params["gamma"],
        "best_C": best_params["C"],
    }
    return selector, pipe, info


# ===================================================================
# Strategy implementations
# ===================================================================

def _score(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray,
           probs_cal: np.ndarray | None = None,
           y_cal: np.ndarray | None = None) -> dict:
    """Score predictions.  If cal probs/labels are provided, also compute
    a personalised threshold via Youden's J and report metrics at that threshold."""
    d = {
        "auc":     _auc(y_true, probs),
        "bal_acc": float(balanced_accuracy_score(y_true, preds)),
        "f1":      float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }
    if probs_cal is not None and y_cal is not None:
        threshold, youden_j = _youden_threshold(y_cal, probs_cal)
        preds_t = (probs >= threshold).astype(int)
        high_mask = y_true == 1
        low_mask  = y_true == 0
        d["threshold"]  = threshold
        d["youden_j"]   = youden_j
        d["bal_acc_t"]  = float(balanced_accuracy_score(y_true, preds_t))
        d["sens_t"] = float(np.mean(preds_t[high_mask] == 1)) if high_mask.any() else float("nan")
        d["spec_t"] = float(np.mean(preds_t[low_mask]  == 0)) if low_mask.any()  else float("nan")
    return d


def _strategy_group_only(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """A: Apply group model directly, ignoring calibration data for training.

    Calibration data is used only to set the classification threshold
    via Youden's J statistic.
    """
    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)
    probs_cal = group_pipe.predict_proba(X_cal_sel)[:, 1]
    probs     = group_pipe.predict_proba(X_test_sel)[:, 1]
    preds     = group_pipe.predict(X_test_sel)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_finetune_head(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """B: Freeze Scaler+Nystroem, refit LogReg head on cal data."""
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]

    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)

    X_cal_nys  = nys.transform(sc.transform(X_cal_sel))
    X_test_nys = nys.transform(sc.transform(X_test_sel))

    clf = LogisticRegression(C=1.0, max_iter=1000,
                             class_weight="balanced", random_state=SEED)
    clf.fit(X_cal_nys, y_cal)
    probs_cal = clf.predict_proba(X_cal_nys)[:, 1]
    probs     = clf.predict_proba(X_test_nys)[:, 1]
    preds     = clf.predict(X_test_nys)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_prior_reg(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """C: Warm-start from group weights with strong L2 (Bayesian prior)."""
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)

    X_cal_nys  = nys.transform(sc.transform(X_cal_sel))
    X_test_nys = nys.transform(sc.transform(X_test_sel))

    # Warm-start with group weights; strong L2 (C=0.01) pulls toward group
    clf = LogisticRegression(C=0.01, max_iter=1000, warm_start=True,
                             class_weight="balanced", random_state=SEED)
    clf.classes_   = group_clf.classes_.copy()
    clf.coef_      = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(X_cal_nys, y_cal)

    probs_cal = clf.predict_proba(X_cal_nys)[:, 1]
    probs     = clf.predict_proba(X_test_nys)[:, 1]
    preds     = clf.predict(X_test_nys)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_scratch_rbf(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """D: Full RBF pipeline from scratch on cal data only."""
    n_cal = X_cal.shape[0]
    # Clamp Nystroem components and K to avoid n > n_samples
    n_comp = min(_RBF_NYS, max(n_cal - 1, 1))
    k = min(_RBF_K, X_cal.shape[1], max(n_cal - 1, 1))

    sel = SelectKBest(f_classif, k=k)
    X_cal_sel  = sel.fit_transform(X_cal, y_cal)
    X_test_sel = sel.transform(X_test)

    pipe = _make_rbf(_RBF_GAMMA, _RBF_C, n_comp, SEED)
    pipe.fit(X_cal_sel, y_cal)
    probs_cal = pipe.predict_proba(X_cal_sel)[:, 1]
    probs     = pipe.predict_proba(X_test_sel)[:, 1]
    preds     = pipe.predict(X_test_sel)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_scratch_logreg(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """E: Plain LogReg from scratch on cal data only."""
    pipe = _make_logreg(C=0.001, seed=SEED)
    pipe.fit(X_cal, y_cal)
    probs_cal = pipe.predict_proba(X_cal)[:, 1]
    probs     = pipe.predict_proba(X_test)[:, 1]
    preds     = pipe.predict(X_test)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_scratch_rf(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """F: Random Forest from scratch on cal data only."""
    pipe = _make_rf(seed=SEED)
    pipe.fit(X_cal, y_cal)
    probs_cal = pipe.predict_proba(X_cal)[:, 1]
    probs     = pipe.predict_proba(X_test)[:, 1]
    preds     = pipe.predict(X_test)
    return _score(y_test, probs, preds, probs_cal, y_cal)


# ===================================================================
# Main experiment loop
# ===================================================================

def run_experiment(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    feat_names: list[str],
    pids: list[str],
    cal_durations: list[int],
    X_by_test: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run the full personalisation comparison.

    If *X_by_test* is provided, the held-out participant's features are
    taken from that dict (e.g. calibration-normalised), while the group
    model trains on *X_by* (pp z-scored).

    Returns a nested dict:  results[pid][cal_dur][strategy] = {auc, bal_acc, f1}
    """
    if X_by_test is None:
        X_by_test = X_by
    results: dict = {}
    n_total = len(pids)

    for i, pid in enumerate(pids):
        t0 = time.time()
        print(f"  [{i+1:>2}/{n_total}] {pid} ...", end=" ", flush=True)

        # 1. Train group model on 27 participants
        selector, group_pipe, group_info = _train_group_model(X_by, y_by, pid)

        results[pid] = {"group_model": group_info}
        # Per-pid RNG so results are reproducible and independent
        rng = np.random.default_rng(SEED + hash(pid) % (2**31))

        for dur in cal_durations:
            # 2. Random cal/test split from all 4 condition blocks
            X_cal, y_cal, X_test, y_test, split_info = _random_cal_split(
                X_by_test[pid], y_by[pid], dur, rng)

            if len(X_cal) < 4 or len(X_test) < 4:
                results[pid][str(dur)] = {"SKIP": f"too few cal={len(X_cal)} test={len(X_test)}"}
                continue
            if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                results[pid][str(dur)] = {"SKIP": "single-class cal or test"}
                continue

            dur_results: dict = {"split_info": split_info}

            # A: group_only (cal used only for threshold, not training)
            dur_results["A_group_only"] = _strategy_group_only(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # B: finetune_head
            dur_results["B_finetune_head"] = _strategy_finetune_head(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # C: prior_reg
            dur_results["C_prior_reg"] = _strategy_prior_reg(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # D: scratch_rbf
            dur_results["D_scratch_rbf"] = _strategy_scratch_rbf(
                X_cal, y_cal, X_test, y_test)

            # E: scratch_logreg
            dur_results["E_scratch_logreg"] = _strategy_scratch_logreg(
                X_cal, y_cal, X_test, y_test)

            # F: scratch_rf
            dur_results["F_scratch_rf"] = _strategy_scratch_rf(
                X_cal, y_cal, X_test, y_test)

            results[pid][str(dur)] = dur_results

        elapsed = time.time() - t0
        # Show group AUC at first available duration as sanity check
        first_dur = str(cal_durations[0])
        a_auc = results[pid].get(first_dur, {}).get("A_group_only", {}).get("auc", float("nan"))
        print(f"group AUC={a_auc:.4f}  k={group_info['best_k']}  [{elapsed:.0f}s]")

    return results


# ===================================================================
# Summary printing
# ===================================================================

_STRATEGIES = [
    "A_group_only", "B_finetune_head", "C_prior_reg",
    "D_scratch_rbf", "E_scratch_logreg", "F_scratch_rf",
]
_SHORT_NAMES = {
    "A_group_only":     "Group",
    "B_finetune_head":  "FT-head",
    "C_prior_reg":      "Prior",
    "D_scratch_rbf":    "Sc-RBF",
    "E_scratch_logreg": "Sc-LR",
    "F_scratch_rf":     "Sc-RF",
}


def _print_summary(results: dict, cal_durations: list[int]) -> None:
    """Print summary tables: mean AUC, median AUC, and split sizes."""
    pids = sorted(p for p in results if p.startswith("P"))

    print()
    print("=" * 90)
    print(f"  PERSONALISATION COMPARISON  (mean AUC +- std, n={len(pids)})")
    print("=" * 90)

    # Mean AUC table
    header = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header += f"  {_SHORT_NAMES[s]:>12}"
    print(header)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            aucs = []
            for pid in pids:
                entry = results[pid].get(str(dur), {}).get(s)
                if entry is not None and isinstance(entry, dict) and "auc" in entry:
                    aucs.append(entry["auc"])
            if aucs:
                m, sd = float(np.mean(aucs)), float(np.std(aucs))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # Median AUC table
    print()
    print(f"  Median AUC (n={len(pids)}):")
    header2 = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header2 += f"  {_SHORT_NAMES[s]:>9}"
    print(header2)
    print("  " + "-" * (6 + 11 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            aucs = []
            for pid in pids:
                entry = results[pid].get(str(dur), {}).get(s)
                if entry is not None and isinstance(entry, dict) and "auc" in entry:
                    aucs.append(entry["auc"])
            if aucs:
                row += f"  {float(np.median(aucs)):>9.4f}"
            else:
                row += f"  {'n/a':>9}"
        print(row)

    # Youden threshold table
    print()
    print(f"  Youden threshold on cal data (mean +- std, n={len(pids)}):")
    header_t = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header_t += f"  {_SHORT_NAMES[s]:>12}"
    print(header_t)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            vals = []
            for pid in pids:
                entry = results[pid].get(str(dur), {}).get(s)
                if entry is not None and isinstance(entry, dict) and "threshold" in entry:
                    vals.append(entry["threshold"])
            if vals:
                m, sd = float(np.mean(vals)), float(np.std(vals))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # BalAcc at personalised threshold
    print()
    print(f"  BalAcc at personalised threshold (mean +- std, n={len(pids)}):")
    header_ba = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header_ba += f"  {_SHORT_NAMES[s]:>12}"
    print(header_ba)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            vals = []
            for pid in pids:
                entry = results[pid].get(str(dur), {}).get(s)
                if entry is not None and isinstance(entry, dict) and "bal_acc_t" in entry:
                    vals.append(entry["bal_acc_t"])
            if vals:
                m, sd = float(np.mean(vals)), float(np.std(vals))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # Sensitivity / Specificity at personalised threshold
    print()
    print(f"  Sensitivity / Specificity at personalised threshold (mean, n={len(pids)}):")
    header_ss = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header_ss += f"  {_SHORT_NAMES[s]:>12}"
    print(header_ss)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            sens_vals, spec_vals = [], []
            for pid in pids:
                entry = results[pid].get(str(dur), {}).get(s)
                if entry is not None and isinstance(entry, dict):
                    if "sens_t" in entry:
                        sens_vals.append(entry["sens_t"])
                    if "spec_t" in entry:
                        spec_vals.append(entry["spec_t"])
            if sens_vals and spec_vals:
                se = float(np.mean(sens_vals))
                sp = float(np.mean(spec_vals))
                row += f"  {se:.2f}/{sp:.2f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # Split size info (from first available PID)
    print()
    print("  Cal / test split sizes (typical, from first participant):")
    first_pid = pids[0] if pids else None
    if first_pid:
        for dur in cal_durations:
            info = results[first_pid].get(str(dur), {}).get("split_info")
            if info:
                print(f"    {dur:>3}s → {info['n_cal']:>3} cal, "
                      f"{info['n_excluded']:>3} gap-excluded, "
                      f"{info['n_test']:>3} test  "
                      f"(blocks: {info['blocks']})")
    print()


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Personalisation vs from-scratch comparison for MWL classification")
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Run only these participant IDs (e.g., --only P05 P07)")
    args = parser.parse_args()

    print("Personalisation vs From-Scratch Comparison")
    print(f"  Dataset    : {args.dataset.name}")
    print(f"  Cal durs   : {_CAL_DURATIONS} s per label")
    print(f"  Gap radius : +-{_GAP_RADIUS} epochs ({_GAP_RADIUS * _STEP_S:.1f} s)")
    print(f"  Sampling   : random within each condition block, equal across blocks")
    print(f"  RBF config : g={_RBF_GAMMA}, C={_RBF_C}, nys={_RBF_NYS}, K={_RBF_K}")
    print(f"  Seed       : {SEED}")
    print()

    # ------------------------------------------------------------------
    # Load features from cache
    # ------------------------------------------------------------------
    key    = _cache_key(args.dataset)
    cached = _load_feature_cache(_FEATURE_CACHE, key)

    if cached is None:
        print("ERROR: Feature cache not found or key mismatch.")
        print("       Run personalised_logreg.py first to build the cache.")
        sys.exit(1)

    X_by, y_by, feat_names = cached
    X_by = {p: v for p, v in X_by.items() if p not in _EXCLUDE}
    y_by = {p: v for p, v in y_by.items() if p not in _EXCLUDE}
    pids = sorted(X_by.keys())
    print(f"Features loaded from cache  ({len(pids)} participants, "
          f"{len(feat_names)} features)  [key={key}]")

    # Per-participant normalisation (mixed: causal for test)
    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    X_by_test: dict[str, np.ndarray] | None = None
    if baseline_by is not None:
        X_by, X_by_test = prepare_mixed_norm(X_by, baseline_by)
        _norm_mode = "mixed_pp_zscore_train_calibration_test"
        print("Mixed normalisation: pp z-score (train) + calibration (test).")
    else:
        for pid in X_by:
            sc_pp = StandardScaler()
            X_by[pid] = sc_pp.fit_transform(X_by[pid])
        _norm_mode = "pp_zscore_full_session"
        print("Per-participant z-scoring applied (baseline cache not found).")
    print()

    # Filter to --only if specified
    if args.only:
        pids = [p for p in pids if p in args.only]
        if not pids:
            print(f"ERROR: None of {args.only} found in included participants.")
            sys.exit(1)
        print(f"Running for {len(pids)} participant(s): {pids}")
        print()

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------
    t_start = time.time()
    results = run_experiment(X_by, y_by, feat_names, pids, _CAL_DURATIONS,
                             X_by_test=X_by_test)
    t_total = (time.time() - t_start) / 60

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(results, _CAL_DURATIONS)
    print(f"  Total wall time: {t_total:.1f} min")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _jsonable(obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_data = {
        "config": {
            "normalisation": _norm_mode,
            "cal_durations_s": _CAL_DURATIONS,
            "gap_radius": _GAP_RADIUS,
            "step_s": _STEP_S,
            "sampling": "random_within_blocks",
            "rbf_gamma": _RBF_GAMMA,
            "rbf_C": _RBF_C,
            "rbf_nys": _RBF_NYS,
            "rbf_K": _RBF_K,
            "n_participants": len(pids),
            "n_features": len(feat_names),
            "seed": SEED,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, default=_jsonable))
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
