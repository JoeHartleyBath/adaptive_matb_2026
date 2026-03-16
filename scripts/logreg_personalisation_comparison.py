"""Personalisation comparison for LogReg MWL classification.

For each of the included participants (LOSO), this script:
  1. Trains a group LogReg model on the other N-1 participants using
     the frozen config from dc_logreg_hyperparameter_plateau (K=30,
     C=0.001, L2, StandardScaler, calibration normalisation).
  2. Splits the held-out participant's data into calibration and test
     sets by randomly sampling from all 4 VR-TSST condition blocks,
     with a gap buffer to prevent window-overlap leakage.
  3. Evaluates six personalisation strategies on the test portion:
       A  group_only      — apply group model directly; cal for threshold only
       B  warm_strong_l2  — warm-start from group weights, C=0.01 (strong prior)
       C  warm_weak_l2    — warm-start from group weights, C=0.1 (more adaptation)
       D  scratch_logreg  — plain LogReg trained from scratch on cal data only
       E  scratch_rf      — Random Forest trained from scratch on cal data only
       F  incremental_sgd — SGDClassifier initialised from group weights,
                            partial_fit on cal data (simulates online update)

Calibration durations (per label): 30 s, 60 s, 90 s, 120 s, 180 s.
Calibration normalisation (fixation + Forest_0 baseline, ADR-0004) is
applied to ALL participants (both train and test) — matching the online
deployment scenario.

Usage:
    python scripts/logreg_personalisation_comparison.py
    python scripts/logreg_personalisation_comparison.py --only P05
    python scripts/logreg_personalisation_comparison.py --out results/test_pretrain/custom.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import (  # noqa: E402
    calibration_norm_features,
    load_baseline_from_cache,
)

_NORM_CACHE  = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_QC_CONFIG   = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_DEFAULT_OUT = _REPO_ROOT / "results" / "test_pretrain" / "logreg_personalisation_comparison.json"

# Frozen LogReg config (from dc_logreg_hyperparameter_plateau)
_LOGREG_K = 30           # SelectKBest k (f_classif)
_LOGREG_C = 0.001        # L2 regularisation
_LOGREG_PENALTY = "l2"
_LOGREG_SCALER  = "standard"

# Warm-start C values bracket the adaptation range
_WARM_C_STRONG = 0.01    # strong prior — stays close to group
_WARM_C_WEAK   = 0.1     # weak prior   — allows more individual shift

# Calibration durations (seconds per label)
_CAL_DURATIONS = [30, 60, 90, 120, 180]
_STEP_S     = 0.5         # epoch step in the HDF5 export
_GAP_RADIUS = 3           # gap zone around cal chunks (in epochs)

SEED = 42


def _load_exclude(cfg_path: Path) -> set[str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_EXCLUDE = _load_exclude(_QC_CONFIG)


# ===================================================================
# Pipeline factories
# ===================================================================

def _make_logreg(C: float = _LOGREG_C, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000,
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
# Scoring helpers
# ===================================================================

def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Optimal threshold via Youden's J.  Returns (threshold, youden_j)."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


def _score(y_true: np.ndarray, probs: np.ndarray, preds: np.ndarray,
           probs_cal: np.ndarray | None = None,
           y_cal: np.ndarray | None = None) -> dict:
    """Score predictions, optionally with personalised Youden threshold."""
    d: dict = {
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


# ===================================================================
# Block detection & random calibration split
# ===================================================================

def _detect_blocks(y: np.ndarray) -> list[dict]:
    """Detect contiguous condition blocks from the label sequence."""
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

    Selects a contiguous chunk per condition block, with gap exclusion
    to prevent window-overlap leakage.
    """
    n_cal_per_label = int(cal_seconds / _STEP_S)
    blocks = _detect_blocks(y)

    blocks_by_label: dict[int, list[dict]] = {}
    for b in blocks:
        blocks_by_label.setdefault(b["label"], []).append(b)

    cal_indices:      list[int] = []
    excluded_indices: set[int]  = set()

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
                    excluded_indices.add(before)
                if after < block["end"]:
                    excluded_indices.add(after)

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
# Group model training (LOSO — frozen LogReg config, no inner CV)
# ===================================================================

def _train_group_logreg(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    held_out_pid: str,
) -> tuple[SelectKBest, Pipeline, dict]:
    """Train the frozen LogReg pipeline on all participants except held_out_pid.

    Config is frozen from dc_logreg_hyperparameter_plateau: K=30, C=0.001,
    L2, StandardScaler.  No inner CV needed (plateau confirmed).
    """
    train_pids = sorted(p for p in X_by if p != held_out_pid)
    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])

    selector = SelectKBest(f_classif, k=_LOGREG_K)
    X_train_sel = selector.fit_transform(X_train, y_train)

    pipe = _make_logreg(C=_LOGREG_C, seed=SEED)
    pipe.fit(X_train_sel, y_train)

    info = {"k": _LOGREG_K, "C": _LOGREG_C, "n_train": len(y_train)}
    return selector, pipe, info


# ===================================================================
# Strategy implementations
# ===================================================================

def _strategy_group_only(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """A: Apply group model directly; cal data for Youden threshold only."""
    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)
    probs_cal = group_pipe.predict_proba(X_cal_sel)[:, 1]
    probs     = group_pipe.predict_proba(X_test_sel)[:, 1]
    preds     = group_pipe.predict(X_test_sel)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_warm_strong_l2(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """B: Warm-start from group weights with strong L2 (C=0.01).

    Freezes the group scaler; refits LogReg head initialised from
    group weights.  Strong L2 pulls the solution toward the group prior.
    """
    sc        = group_pipe.named_steps["sc"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)

    X_cal_sc  = sc.transform(X_cal_sel)
    X_test_sc = sc.transform(X_test_sel)

    clf = LogisticRegression(C=_WARM_C_STRONG, max_iter=2000,
                             warm_start=True, class_weight="balanced",
                             random_state=SEED)
    clf.classes_   = group_clf.classes_.copy()
    clf.coef_      = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(X_cal_sc, y_cal)

    probs_cal = clf.predict_proba(X_cal_sc)[:, 1]
    probs     = clf.predict_proba(X_test_sc)[:, 1]
    preds     = clf.predict(X_test_sc)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_warm_weak_l2(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """C: Warm-start from group weights with weak L2 (C=0.1).

    Same as B but weaker regularisation allows more individual adaptation.
    """
    sc        = group_pipe.named_steps["sc"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)

    X_cal_sc  = sc.transform(X_cal_sel)
    X_test_sc = sc.transform(X_test_sel)

    clf = LogisticRegression(C=_WARM_C_WEAK, max_iter=2000,
                             warm_start=True, class_weight="balanced",
                             random_state=SEED)
    clf.classes_   = group_clf.classes_.copy()
    clf.coef_      = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(X_cal_sc, y_cal)

    probs_cal = clf.predict_proba(X_cal_sc)[:, 1]
    probs     = clf.predict_proba(X_test_sc)[:, 1]
    preds     = clf.predict(X_test_sc)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_scratch_logreg(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """D: LogReg from scratch on cal data only."""
    n_cal = X_cal.shape[0]
    k = min(_LOGREG_K, X_cal.shape[1], max(n_cal - 1, 1))

    sel = SelectKBest(f_classif, k=k)
    X_cal_sel  = sel.fit_transform(X_cal, y_cal)
    X_test_sel = sel.transform(X_test)

    pipe = _make_logreg(C=_LOGREG_C, seed=SEED)
    pipe.fit(X_cal_sel, y_cal)
    probs_cal = pipe.predict_proba(X_cal_sel)[:, 1]
    probs     = pipe.predict_proba(X_test_sel)[:, 1]
    preds     = pipe.predict(X_test_sel)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_scratch_rf(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """E: Random Forest from scratch on cal data only."""
    pipe = _make_rf(seed=SEED)
    pipe.fit(X_cal, y_cal)
    probs_cal = pipe.predict_proba(X_cal)[:, 1]
    probs     = pipe.predict_proba(X_test)[:, 1]
    preds     = pipe.predict(X_test)
    return _score(y_test, probs, preds, probs_cal, y_cal)


def _strategy_incremental_sgd(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """F: SGDClassifier initialised from group LogReg, partial_fit on cal.

    Simulates an online-style incremental update.  The SGD loss='log_loss'
    matches logistic regression; weights are seeded from the group model.
    """
    sc        = group_pipe.named_steps["sc"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel  = selector.transform(X_cal)
    X_test_sel = selector.transform(X_test)

    X_cal_sc  = sc.transform(X_cal_sel).astype(np.float64)
    X_test_sc = sc.transform(X_test_sel).astype(np.float64)

    sgd = SGDClassifier(
        loss="log_loss",
        learning_rate="adaptive",
        eta0=0.01,
        class_weight="balanced",
        random_state=SEED,
        warm_start=True,
    )
    # Initialise from group model weights
    sgd.classes_   = group_clf.classes_.copy()
    sgd.coef_      = group_clf.coef_.copy()
    sgd.intercept_ = group_clf.intercept_.copy()
    sgd.t_         = 1.0  # reset iteration counter for learning rate schedule
    sgd.partial_fit(X_cal_sc, y_cal, classes=group_clf.classes_)

    # SGD decision_function → sigmoid for probabilities
    raw = sgd.decision_function(X_test_sc)
    probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
    raw_cal = sgd.decision_function(X_cal_sc)
    probs_cal = 1.0 / (1.0 + np.exp(-np.clip(raw_cal, -500, 500)))

    preds = (probs >= 0.5).astype(int)
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
) -> dict:
    """Run the full LOSO personalisation comparison.

    Returns nested dict:  results[pid][cal_dur][strategy] = {auc, ...}
    """
    results: dict = {}
    n_total = len(pids)

    for i, pid in enumerate(pids):
        t0 = time.time()
        print(f"  [{i+1:>2}/{n_total}] {pid} ...", end=" ", flush=True)

        selector, group_pipe, group_info = _train_group_logreg(X_by, y_by, pid)
        results[pid] = {"group_model": group_info}

        rng = np.random.default_rng(SEED + hash(pid) % (2**31))

        for dur in cal_durations:
            X_cal, y_cal, X_test, y_test, split_info = _random_cal_split(
                X_by[pid], y_by[pid], dur, rng)

            if len(X_cal) < 4 or len(X_test) < 4:
                results[pid][str(dur)] = {
                    "SKIP": f"too few cal={len(X_cal)} test={len(X_test)}"
                }
                continue
            if len(np.unique(y_cal)) < 2 or len(np.unique(y_test)) < 2:
                results[pid][str(dur)] = {"SKIP": "single-class cal or test"}
                continue

            dur_results: dict = {"split_info": split_info}

            # A: group_only
            dur_results["A_group_only"] = _strategy_group_only(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # B: warm_strong_l2
            dur_results["B_warm_strong_l2"] = _strategy_warm_strong_l2(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # C: warm_weak_l2
            dur_results["C_warm_weak_l2"] = _strategy_warm_weak_l2(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            # D: scratch_logreg
            dur_results["D_scratch_logreg"] = _strategy_scratch_logreg(
                X_cal, y_cal, X_test, y_test)

            # E: scratch_rf
            dur_results["E_scratch_rf"] = _strategy_scratch_rf(
                X_cal, y_cal, X_test, y_test)

            # F: incremental_sgd
            dur_results["F_incremental_sgd"] = _strategy_incremental_sgd(
                selector, group_pipe, X_cal, y_cal, X_test, y_test)

            results[pid][str(dur)] = dur_results

        elapsed = time.time() - t0
        first_dur = str(cal_durations[0])
        a_auc = results[pid].get(first_dur, {}).get(
            "A_group_only", {}).get("auc", float("nan"))
        print(f"group AUC={a_auc:.4f}  [{elapsed:.1f}s]")

    return results


# ===================================================================
# Summary printing
# ===================================================================

_STRATEGIES = [
    "A_group_only", "B_warm_strong_l2", "C_warm_weak_l2",
    "D_scratch_logreg", "E_scratch_rf", "F_incremental_sgd",
]
_SHORT_NAMES = {
    "A_group_only":       "Group",
    "B_warm_strong_l2":   "WS-strong",
    "C_warm_weak_l2":     "WS-weak",
    "D_scratch_logreg":   "Sc-LR",
    "E_scratch_rf":       "Sc-RF",
    "F_incremental_sgd":  "SGD-inc",
}


def _print_summary(results: dict, cal_durations: list[int]) -> None:
    pids = sorted(p for p in results if p.startswith("P"))

    print()
    print("=" * 96)
    print(f"  LOGREG PERSONALISATION COMPARISON  (mean AUC +- std, n={len(pids)})")
    print("=" * 96)

    # --- Mean AUC ---
    header = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header += f"  {_SHORT_NAMES[s]:>12}"
    print(header)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            aucs = [
                results[p].get(str(dur), {}).get(s, {}).get("auc")
                for p in pids
            ]
            aucs = [a for a in aucs if a is not None and not np.isnan(a)]
            if aucs:
                m, sd = float(np.mean(aucs)), float(np.std(aucs))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # --- Median AUC ---
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
            aucs = [
                results[p].get(str(dur), {}).get(s, {}).get("auc")
                for p in pids
            ]
            aucs = [a for a in aucs if a is not None and not np.isnan(a)]
            row += f"  {float(np.median(aucs)):>9.4f}" if aucs else f"  {'n/a':>9}"
        print(row)

    # --- Delta vs Group ---
    print()
    print(f"  Delta AUC vs Group (mean, n={len(pids)}):")
    header_d = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES[1:]:  # skip group_only
        header_d += f"  {_SHORT_NAMES[s]:>12}"
    print(header_d)
    print("  " + "-" * (6 + 14 * (len(_STRATEGIES) - 1)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES[1:]:
            deltas = []
            for p in pids:
                s_auc = results[p].get(str(dur), {}).get(s, {}).get("auc")
                g_auc = results[p].get(str(dur), {}).get("A_group_only", {}).get("auc")
                if s_auc is not None and g_auc is not None and not (
                    np.isnan(s_auc) or np.isnan(g_auc)
                ):
                    deltas.append(s_auc - g_auc)
            if deltas:
                m = float(np.mean(deltas))
                row += f"  {m:>+11.4f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # --- Youden threshold ---
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
            vals = [
                results[p].get(str(dur), {}).get(s, {}).get("threshold")
                for p in pids
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                m, sd = float(np.mean(vals)), float(np.std(vals))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # --- BalAcc at personalised threshold ---
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
            vals = [
                results[p].get(str(dur), {}).get(s, {}).get("bal_acc_t")
                for p in pids
            ]
            vals = [v for v in vals if v is not None]
            if vals:
                m, sd = float(np.mean(vals)), float(np.std(vals))
                row += f"  {m:>5.3f}+-{sd:.3f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # --- Sensitivity / Specificity ---
    print()
    print(f"  Sens / Spec at personalised threshold (mean, n={len(pids)}):")
    header_ss = f"  {'Cal(s)':>6}"
    for s in _STRATEGIES:
        header_ss += f"  {_SHORT_NAMES[s]:>12}"
    print(header_ss)
    print("  " + "-" * (6 + 14 * len(_STRATEGIES)))

    for dur in cal_durations:
        row = f"  {dur:>6}"
        for s in _STRATEGIES:
            sens_vals = [
                results[p].get(str(dur), {}).get(s, {}).get("sens_t")
                for p in pids
            ]
            spec_vals = [
                results[p].get(str(dur), {}).get(s, {}).get("spec_t")
                for p in pids
            ]
            sens_vals = [v for v in sens_vals if v is not None and not np.isnan(v)]
            spec_vals = [v for v in spec_vals if v is not None and not np.isnan(v)]
            if sens_vals and spec_vals:
                se = float(np.mean(sens_vals))
                sp = float(np.mean(spec_vals))
                row += f"  {se:.2f}/{sp:.2f}"
            else:
                row += f"  {'n/a':>12}"
        print(row)

    # --- Split sizes ---
    print()
    print("  Cal / test split sizes (typical, from first participant):")
    first_pid = pids[0] if pids else None
    if first_pid:
        for dur in cal_durations:
            info = results[first_pid].get(str(dur), {}).get("split_info")
            if info:
                print(f"    {dur:>3}s -> {info['n_cal']:>3} cal, "
                      f"{info['n_excluded']:>3} gap-excluded, "
                      f"{info['n_test']:>3} test  "
                      f"(blocks: {info['blocks']})")
    print()


# ===================================================================
# Data loading
# ===================================================================

def load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                          dict[str, np.ndarray] | None,
                          list[str], list[str]]:
    """Load calibration-normalised features from the norm cache.

    Returns (X_by, y_by, stress_y_by, feat_names, pids).
    stress_y_by is None when the cache predates stress label support.
    """
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
    has_stress = f"{available[0]}_task_stress_y" in npz
    stress_y_by: dict[str, np.ndarray] | None = {} if has_stress else None
    for pid in available:
        X_by_raw[pid] = npz[f"{pid}_task_X"]
        y_by[pid] = npz[f"{pid}_task_y"]
        if has_stress:
            stress_y_by[pid] = npz[f"{pid}_task_stress_y"]  # type: ignore[index]

    # Calibration normalisation (ADR-0004) for ALL participants
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
    return X_by, y_by, stress_y_by, feat_names, pids


# ===================================================================
# Entry point
# ===================================================================

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="LogReg personalisation comparison for MWL classification")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Run only these participant IDs (e.g., --only P05 P07)")
    parser.add_argument("--stress", action="store_true",
                        help="Classify stress (High/Low) instead of cognitive load.")
    args = parser.parse_args()

    print("LogReg Personalisation Comparison")
    print(f"  Frozen config  : K={_LOGREG_K}, C={_LOGREG_C}, {_LOGREG_PENALTY}, "
          f"{_LOGREG_SCALER}")
    print(f"  Cal durations  : {_CAL_DURATIONS} s per label")
    print(f"  Gap radius     : +-{_GAP_RADIUS} epochs ({_GAP_RADIUS * _STEP_S:.1f} s)")
    print(f"  Warm-start C   : strong={_WARM_C_STRONG}, weak={_WARM_C_WEAK}")
    print(f"  Normalisation  : calibration (fixation + Forest_0)")
    print(f"  Seed           : {SEED}")
    print()

    # Load data
    X_by, y_by_cog, stress_y_by, feat_names, pids = load_data()

    if args.stress:
        if stress_y_by is None:
            raise SystemExit(
                "ERROR: Stress labels not found in norm cache.\n"
                "       Rebuild the cache: python scripts/causal_norm_comparison.py"
            )
        y_by = stress_y_by
        target_name = "stress"
    else:
        y_by = y_by_cog
        target_name = "cognitive_load"

    print(f"  Target         : {target_name}")
    print(f"  Participants   : {len(pids)}")
    print(f"  Features       : {len(feat_names)}")
    print()

    if args.only:
        pids = [p for p in pids if p in args.only]
        if not pids:
            print(f"ERROR: None of {args.only} found in included participants.")
            sys.exit(1)
        print(f"Running for {len(pids)} participant(s): {pids}")
        print()

    # Run experiment
    t_start = time.time()
    results = run_experiment(X_by, y_by, feat_names, pids, _CAL_DURATIONS)
    t_total = (time.time() - t_start) / 60

    # Summary
    _print_summary(results, _CAL_DURATIONS)
    print(f"  Total wall time: {t_total:.1f} min")

    # Save
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
            "normalisation": "calibration_norm_fixation_forest0",
            "cal_durations_s": _CAL_DURATIONS,
            "gap_radius": _GAP_RADIUS,
            "step_s": _STEP_S,
            "sampling": "random_within_blocks",
            "logreg_K": _LOGREG_K,
            "logreg_C": _LOGREG_C,
            "logreg_penalty": _LOGREG_PENALTY,
            "warm_C_strong": _WARM_C_STRONG,
            "warm_C_weak": _WARM_C_WEAK,
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
