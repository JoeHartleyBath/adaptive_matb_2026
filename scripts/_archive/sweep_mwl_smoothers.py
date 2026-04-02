"""sweep_mwl_smoothers.py

Sweep smoother hyperparameters, calibration durations, and hysteresis margins
across all included participants using LOSO.  Reuses group-model and
calibration infrastructure from simulate_mwl_adaptation.py.

Usage:
    python scripts/sweep_mwl_smoothers.py
    python scripts/sweep_mwl_smoothers.py --only P05 --jobs 1
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Repo root + src on path
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptation.mwl_smoother import (                                    # noqa: E402
    MwlSmootherConfig,
    EmaSmoother,
    SmaSmoother,
    AdaptiveEmaSmoother,
    FixedLagSmoother,
)
from ml.pretrain_loader import load_baseline_from_cache, prepare_mixed_norm  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & constants — must match personalisation_comparison.py exactly
# ---------------------------------------------------------------------------

_DATASET       = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_FEATURE_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "feature_cache.npz"
_NORM_CACHE    = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_DEFAULT_OUT   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep.json"
_CSV_PID       = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep.csv"
_CSV_SUMMARY   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep_summary.csv"
_FIG_DIR       = _REPO_ROOT / "results" / "figures" / "smoother_sweep"
_GROUP_CACHE   = _REPO_ROOT / "results" / "test_pretrain" / "group_model_cache"

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

# RBF frozen config (from DC-08 ablation) — must match personalisation_comparison.py
_RBF_GAMMA = 0.01
_RBF_C     = 1.0
_RBF_NYS   = 300
_RBF_K     = 30
_N_INNER   = 5

# Simulation parameters
_CAL_DURATION_S   = 30     # seconds of cal data per label
_STEP_S           = 0.5    # epoch step in seconds (matches HDF5 export)
_GAP_RADIUS       = 3      # epochs excluded around each cal chunk boundary
SEED = 42


# ===========================================================================
# Pipeline factories (must match personalisation_comparison.py)
# ===========================================================================

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


# ===========================================================================
# AUC / threshold helpers
# ===========================================================================

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


# ===========================================================================
# Group model training (LOSO — train on 27, produce fitted pipeline)
# Copied verbatim from personalisation_comparison.py
# ===========================================================================

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


def _load_or_train_group_model(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    held_out_pid: str,
    dataset_key: str,
) -> tuple[SelectKBest, Pipeline, dict]:
    """Load a cached group model or train and cache a new one."""
    import joblib
    cache_path = _GROUP_CACHE / f"{dataset_key}_{held_out_pid}.pkl"
    if cache_path.exists():
        return joblib.load(cache_path)
    selector, pipe, info = _train_group_model(X_by, y_by, held_out_pid)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((selector, pipe, info), cache_path, compress=3)
    return selector, pipe, info


# ===========================================================================
# Block detection & random calibration split
# Copied verbatim from personalisation_comparison.py
# ===========================================================================

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


def _get_test_indices(
    y: np.ndarray,
    cal_seconds: float,
    rng_seed: int,
) -> np.ndarray:
    """Return sorted array of test-epoch indices for a given cal split.

    Re-derives the cal/gap indices using the same RNG seed that was used
    for the actual split, then returns all remaining indices.
    """
    n_cal_per_label = int(cal_seconds / _STEP_S)
    blocks = _detect_blocks(y)
    rng = np.random.default_rng(rng_seed)

    blocks_by_label: dict[int, list[dict]] = {}
    for b in blocks:
        blocks_by_label.setdefault(b["label"], []).append(b)

    cal_indices: list[int] = []
    excluded_indices: set[int] = set()

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
            if max_start <= 0:
                offset = 0
            else:
                offset = int(rng.integers(0, max_start + 1))

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

    cal_set     = set(cal_indices)
    all_indices = set(range(len(y)))
    return np.array(sorted(all_indices - cal_set - excluded_indices), dtype=np.intp)


# ===========================================================================
# FT-head personalisation
# ===========================================================================

def _fit_fthead(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit a personalised LogReg head on Nystroem-projected calibration data.

    Projects X_cal through selector -> StandardScaler -> Nystroem (from the
    group pipeline), then fits a new LogisticRegression on the projected
    features.

    Returns:
        (fitted_clf, probs_cal)  — cal-set predicted probabilities for
        downstream threshold computation.
    """
    X_cal_sel = selector.transform(X_cal)
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]
    X_projected = nys.transform(sc.transform(X_cal_sel))

    clf = LogisticRegression(C=1.0, max_iter=1000,
                             class_weight="balanced",
                             random_state=SEED)
    clf.fit(X_projected, y_cal)

    probs_cal = clf.predict_proba(X_projected)[:, 1]

    return clf, probs_cal


# ===========================================================================
# Smoother factory & quality metrics
# ===========================================================================

def _make_smoother(cfg: MwlSmootherConfig):
    """Instantiate a smoother from a config."""
    if cfg.method == "ema":
        return EmaSmoother(alpha=cfg.alpha)
    if cfg.method == "sma":
        return SmaSmoother(window_n=cfg.window_n)
    if cfg.method == "adaptive_ema":
        return AdaptiveEmaSmoother(
            alpha_min=cfg.alpha_min,
            alpha_max=cfg.alpha_max,
        )
    if cfg.method == "fixed_lag":
        return FixedLagSmoother(
            lag_n=cfg.lag_n,
            process_noise=cfg.process_noise,
            measurement_noise=cfg.measurement_noise,
        )
    raise ValueError(f"Unknown smoother method: {cfg.method}")


def _compute_smoother_stats(
    assist_on: np.ndarray,
    y_full: np.ndarray,
    test_indices: np.ndarray | None = None,
) -> dict:
    """Compute quality metrics for one smoother's assist_on predictions.

    Parameters
    ----------
    assist_on : bool array, one per epoch
    y_full : int array, actual MWL labels (0=LOW, 1=HIGH)
    test_indices : optional int array of held-out epoch indices.
        When provided, accuracy metrics (bal_acc, pct_on_hi, pct_off_lo,
        onset_latencies) are computed only on these epochs to avoid
        cal-data leakage.  Switching and bout metrics are always computed
        on the full trace (they reflect deployment behaviour).

    Returns dict with:
        pct_on_hi        % of HIGH epochs where assist is ON
        pct_off_lo       % of LOW epochs where assist is OFF
        bal_acc          balanced accuracy treating assist_on as classifier
        switch_rate_pm   ON<->OFF transitions per minute
        median_bout_s    median duration of contiguous ON or OFF bouts (s)
        onset_latencies  list of seconds until first ON in each HIGH block
    """
    blocks = _detect_blocks(y_full)

    # For accuracy metrics: restrict to test indices if provided
    if test_indices is not None:
        test_set = set(test_indices)
        hi_epochs = np.array([i for b in blocks if b["label"] == 1
                              for i in range(b["start"], b["end"]) if i in test_set])
        lo_epochs = np.array([i for b in blocks if b["label"] == 0
                              for i in range(b["start"], b["end"]) if i in test_set])
    else:
        hi_epochs = np.concatenate([np.arange(b["start"], b["end"])
                                    for b in blocks if b["label"] == 1])
        lo_epochs = np.concatenate([np.arange(b["start"], b["end"])
                                    for b in blocks if b["label"] == 0])

    pct_on_hi  = 100.0 * assist_on[hi_epochs].mean() if len(hi_epochs) else float("nan")
    pct_off_lo = 100.0 * (~assist_on[lo_epochs]).mean() if len(lo_epochs) else float("nan")

    # Balanced accuracy: assist_on as binary prediction of label==1
    sens = assist_on[hi_epochs].mean() if len(hi_epochs) else 0.0
    spec = (~assist_on[lo_epochs]).mean() if len(lo_epochs) else 0.0
    bal_acc = 100.0 * (sens + spec) / 2.0

    # Switch rate: ON<->OFF transitions per minute
    transitions = int(np.sum(np.diff(assist_on.astype(int)) != 0))
    total_time_min = len(assist_on) * _STEP_S / 60.0
    switch_rate_pm = transitions / total_time_min if total_time_min > 0 else 0.0

    # Median bout duration: contiguous runs of same state
    bout_lengths: list[float] = []
    if len(assist_on) > 0:
        run_start = 0
        for i in range(1, len(assist_on)):
            if assist_on[i] != assist_on[run_start]:
                bout_lengths.append((i - run_start) * _STEP_S)
                run_start = i
        bout_lengths.append((len(assist_on) - run_start) * _STEP_S)
    median_bout_s = float(np.median(bout_lengths)) if bout_lengths else 0.0

    # Onset latency: in each HIGH block, seconds until first ON
    # (computed on full trace — onset detection is a deployment metric)
    onset_latencies: list[float] = []
    for b in blocks:
        if b["label"] != 1:
            continue
        b_on = assist_on[b["start"]:b["end"]]
        on_indices = np.where(b_on)[0]
        if len(on_indices) > 0:
            onset_latencies.append(float(on_indices[0] * _STEP_S))
        else:
            onset_latencies.append(float("nan"))

    return {
        "pct_on_hi":       round(pct_on_hi, 1),
        "pct_off_lo":      round(pct_off_lo, 1),
        "bal_acc":         round(bal_acc, 1),
        "switch_rate_pm":  round(switch_rate_pm, 1),
        "median_bout_s":   round(median_bout_s, 1),
        "onset_latencies": [round(x, 1) if not np.isnan(x) else None
                            for x in onset_latencies],
    }


# ===========================================================================
# Cache loading (must match personalised_logreg.py / personalisation_comparison.py)
# ===========================================================================

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


# ===========================================================================
# Per-smoother simulation (with hysteresis)
# ===========================================================================

def _simulate_one_smoother(
    p_high_all: np.ndarray,
    y_full: np.ndarray,
    threshold: float,
    smoother_cfg: MwlSmootherConfig,
    hysteresis_margin: float = 0.0,
) -> np.ndarray:
    """Simulate assistance switching with stateful hysteresis.

    Returns bool array of assist_on decisions (one per epoch).
    """
    smoother = _make_smoother(smoother_cfg)
    n_epochs = len(y_full)
    assist_on = np.zeros(n_epochs, dtype=bool)
    state_on = False

    for i in range(n_epochs):
        smoothed = smoother.update(float(p_high_all[i]))
        if state_on:
            if smoothed < threshold - hysteresis_margin:
                state_on = False
        else:
            if smoothed >= threshold + hysteresis_margin:
                state_on = True
        assist_on[i] = state_on

    return assist_on


# ===========================================================================
# Sweep grid
# ===========================================================================

_CAL_DURATIONS = [60, 120]        # seconds per label (30 s never competitive)
_SEEDS         = [0, 1, 2, 3, 4]
_HYST_MARGINS  = [0.00, 0.02, 0.05, 0.08]

# Threshold strategies (7 total — pruned from 14 based on first sweep)
# 1) Youden + offset: sensitivity analysis around the data-driven optimum
_YOUDEN_OFFSETS = [-0.05, 0.00, +0.05]
# 2) Fixed probability thresholds: model-agnostic baselines
_FIXED_THRESHOLDS = [0.50, 0.60]
# 3) Cost-sensitive Youden: J = w·TPR - (1-w)·FPR, higher w penalises
#    missed HIGH workload more than unnecessary assistance
_COST_WEIGHTS = [0.6, 0.7]


def _compute_thresholds(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
) -> list[tuple[str, float]]:
    """Return list of (strategy_id, threshold) for all 7 strategies.

    Uses calibration labels and predicted probabilities to derive
    data-driven thresholds (Youden, cost-weighted Youden) and also
    includes fixed probability baselines.
    """
    base_thresh, _ = _youden_threshold(y_cal, probs_cal)

    results: list[tuple[str, float]] = []

    # Youden + offset
    for off in _YOUDEN_OFFSETS:
        t = np.clip(base_thresh + off, 0.01, 0.99)
        label = f"youden{off:+.2f}" if off != 0.0 else "youden"
        results.append((label, float(t)))

    # Fixed probability thresholds
    for t in _FIXED_THRESHOLDS:
        results.append((f"fixed_{t:.2f}", t))

    # Cost-sensitive Youden: J = w·TPR - (1-w)·FPR
    if len(np.unique(y_cal)) >= 2:
        fpr, tpr, thresholds = roc_curve(y_cal, probs_cal)
        for w in _COST_WEIGHTS:
            j_w = w * tpr - (1.0 - w) * fpr
            best_idx = int(np.argmax(j_w))
            results.append((f"cost_w{w:.1f}", float(thresholds[best_idx])))
    else:
        for w in _COST_WEIGHTS:
            results.append((f"cost_w{w:.1f}", 0.5))

    return results


def _build_smoother_grid() -> list[tuple[str, MwlSmootherConfig, dict]]:
    """Returns list of (config_id, smoother_cfg, param_dict) tuples."""
    configs: list[tuple[str, MwlSmootherConfig, dict]] = []

    # EMA: keep alpha values that appeared in top-100 (drop 0.15–0.50)
    for alpha in [0.03, 0.05, 0.08, 0.10]:
        cfg = MwlSmootherConfig(method="ema", alpha=alpha)
        configs.append((f"ema_a{alpha}", cfg, {"alpha": alpha}))

    # SMA: representative window sizes (all nearly identical; keep 3)
    for w in [10, 12, 16]:
        cfg = MwlSmootherConfig(method="sma", window_n=w)
        configs.append((f"sma_w{w}", cfg, {"window_n": w}))

    # AdaptiveEMA: drop alpha_min=0.10 (weaker) and alpha_max=0.50 (redundant)
    for amin, amax in itertools.product(
        [0.03, 0.05], [0.15, 0.20, 0.30]
    ):
        if amin >= amax:
            continue
        cfg = MwlSmootherConfig(method="adaptive_ema", alpha_min=amin, alpha_max=amax)
        configs.append((f"aema_{amin}_{amax}", cfg, {"alpha_min": amin, "alpha_max": amax}))

    # FixedLag: only pn=0.001, mn∈{0.1,0.2}, lag∈{2,4,8}
    for lag, pn, mn in itertools.product(
        [2, 4, 8], [0.001], [0.10, 0.20]
    ):
        cfg = MwlSmootherConfig(method="fixed_lag", lag_n=lag,
                                process_noise=pn, measurement_noise=mn)
        configs.append((f"fl_l{lag}_p{pn}_m{mn}", cfg,
                        {"lag_n": lag, "process_noise": pn, "measurement_noise": mn}))

    return configs


# ===========================================================================
# Per-participant worker (parallelisable)
# ===========================================================================

def _process_participant(
    pid: str,
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    X_by_test: dict[str, np.ndarray],
    dataset_key: str,
    smoother_grid: list[tuple[str, MwlSmootherConfig, dict]],
) -> list[dict]:
    """Returns list of flat result dicts — one per (seed, cal_dur, smoother, hysteresis)."""
    selector, group_pipe, _ = _load_or_train_group_model(X_by, y_by, pid, dataset_key)
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]

    # Pre-project all epochs (expensive — do once)
    X_full     = X_by_test[pid]
    y_full     = y_by[pid]
    X_sel      = selector.transform(X_full)
    X_proj_all = nys.transform(sc.transform(X_sel))  # (N, n_components)

    rows: list[dict] = []
    pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)

    for seed in _SEEDS:
        for cal_dur in _CAL_DURATIONS:
            rng = np.random.default_rng(SEED + pid_hash + seed * 1000)
            X_cal, y_cal, X_test, y_test, split_info = _random_cal_split(
                X_full, y_full, cal_dur, rng)

            clf, probs_cal = _fit_fthead(selector, group_pipe, X_cal, y_cal)

            # FT-head AUC
            X_test_sel  = selector.transform(X_test)
            probs_test  = clf.predict_proba(nys.transform(sc.transform(X_test_sel)))[:, 1]
            ft_auc      = _auc(y_test, probs_test)

            # Pre-compute raw probabilities for the full session
            p_high_all = clf.predict_proba(X_proj_all)[:, 1]

            # Derive test indices: re-run split with same rng seed
            test_indices = _get_test_indices(
                y_full, cal_dur, SEED + pid_hash + seed * 1000)

            # All 14 threshold strategies from cal data
            thresh_list = _compute_thresholds(y_cal, probs_cal)

            for thresh_id, threshold in thresh_list:
                for cfg_id, smoother_cfg, param_dict in smoother_grid:
                    for hyst in _HYST_MARGINS:
                        assist_on = _simulate_one_smoother(
                            p_high_all, y_full, threshold, smoother_cfg, hyst)

                        stats = _compute_smoother_stats(assist_on, y_full,
                                                       test_indices=test_indices)
                        _onsets = [x for x in stats["onset_latencies"]
                                  if x is not None]
                        mean_onset = float(np.nanmean(_onsets)) \
                            if _onsets else float("nan")

                        rows.append({
                            "pid":           pid,
                            "seed":          seed,
                            "cal_dur_s":     cal_dur,
                            "thresh_strategy": thresh_id,
                            "config_id":     cfg_id,
                            "method":        smoother_cfg.method,
                            **param_dict,
                            "hysteresis":    hyst,
                            "threshold":     round(threshold, 4),
                            "ft_auc":        round(ft_auc, 4),
                            "bal_acc":       stats["bal_acc"],
                            "pct_on_hi":     stats["pct_on_hi"],
                            "pct_off_lo":    stats["pct_off_lo"],
                            "switch_rate_pm": stats["switch_rate_pm"],
                            "median_bout_s": stats["median_bout_s"],
                            "mean_onset_s":  round(mean_onset, 2),
                        })

    return rows


# ===========================================================================
# Running progress display
# ===========================================================================

def _print_running_best(
    all_rows: list[dict],
    done_pids: list[str],
    elapsed_min: float,
    total_pids: int,
) -> None:
    """Print top-5 configs by mean bal_acc so far."""
    from collections import defaultdict

    # Aggregate by (thresh_strategy, config_id, hysteresis)
    groups: dict[tuple, list[float]] = defaultdict(list)
    sw_groups: dict[tuple, list[float]] = defaultdict(list)
    for r in all_rows:
        key = (r["thresh_strategy"], r["config_id"], r["hysteresis"])
        groups[key].append(r["bal_acc"])
        sw_groups[key].append(r["switch_rate_pm"])

    ranked = []
    for (thresh_id, cfg_id, hyst), vals in groups.items():
        clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        if clean:
            ranked.append((thresh_id, cfg_id, hyst,
                           float(np.mean(clean)), float(np.std(clean)),
                           float(np.mean(sw_groups[(thresh_id, cfg_id, hyst)]))))
    ranked.sort(key=lambda x: x[3], reverse=True)

    n_done = len(done_pids)
    print(f"\n{'='*80}")
    print(f"  {n_done}/{total_pids} participants  |  {elapsed_min:.1f} min  |  "
          f"{len(all_rows):,} rows so far")
    print(f"  Latest: {', '.join(done_pids[-4:])}")
    print(f"  {'─'*76}")
    print(f"  {'Rank':<5} {'Threshold':<14} {'Config':<25} {'Hyst':>5} {'BA%':>7} {'±':>6} {'sw/m':>6}")
    print(f"  {'─'*76}")
    for i, (thresh_id, cfg_id, hyst, mean_ba, std_ba, mean_sw) in enumerate(ranked[:5], 1):
        print(f"  {i:<5} {thresh_id:<14} {cfg_id:<25} {hyst:>5.2f} {mean_ba:>7.1f} {std_ba:>6.1f} {mean_sw:>6.1f}")
    print(f"{'='*80}\n")


# ===========================================================================
# main
# ===========================================================================

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Sweep smoother hyperparameters across all participants (LOSO)")
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument("--out",     type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only",    type=str,  nargs="*", default=None)
    parser.add_argument("--jobs",    type=int,  default=4)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load features from cache
    # ------------------------------------------------------------------
    key    = _cache_key(args.dataset)
    cached = _load_feature_cache(_FEATURE_CACHE, key)
    if cached is None:
        print("ERROR: Feature cache missing — run personalised_logreg.py first")
        sys.exit(1)

    X_by, y_by, _ = cached
    X_by = {p: v for p, v in X_by.items() if p not in _EXCLUDE}
    y_by = {p: v for p, v in y_by.items() if p not in _EXCLUDE}
    pids = sorted(X_by.keys())
    if args.only:
        pids = [p for p in pids if p in args.only]

    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    if baseline_by is not None:
        X_by, X_by_test = prepare_mixed_norm(X_by, baseline_by)
    else:
        for pid in X_by:
            X_by[pid] = StandardScaler().fit_transform(X_by[pid])
        X_by_test = X_by

    smoother_grid = _build_smoother_grid()
    n_thresh = len(_YOUDEN_OFFSETS) + len(_FIXED_THRESHOLDS) + len(_COST_WEIGHTS)
    total = (len(pids) * len(_SEEDS) * len(_CAL_DURATIONS)
             * n_thresh * len(smoother_grid) * len(_HYST_MARGINS))
    print(f"{len(pids)} pids × {len(_SEEDS)} seeds × {len(_CAL_DURATIONS)} cal × "
          f"{n_thresh} thresh × {len(smoother_grid)} smoothers × "
          f"{len(_HYST_MARGINS)} hyst = {total:,} runs")

    # ------------------------------------------------------------------
    # Parallel participant loop (batched for running updates)
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    t0 = time.time()
    batch_size = max(args.jobs, 1)
    done_pids: list[str] = []

    for batch_start in range(0, len(pids), batch_size):
        batch_pids = pids[batch_start:batch_start + batch_size]
        batch_results = Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(_process_participant)(
                pid, X_by, y_by, X_by_test, key, smoother_grid
            ) for pid in batch_pids
        )
        for rows in batch_results:
            all_rows.extend(rows)
        done_pids.extend(batch_pids)

        # --- Running update: top-5 configs so far ---
        elapsed = (time.time() - t0) / 60
        _print_running_best(all_rows, done_pids, elapsed, len(pids))

    print(f"Done in {(time.time()-t0)/60:.1f} min  — {len(all_rows):,} rows")

    # --- CSV output ---
    _write_csv(all_rows, args)

    # --- Summary CSV (mean across seeds and pids) ---
    _write_summary(all_rows, args)

    # --- Heatmaps ---
    _plot_heatmaps(all_rows, args)

    # --- Best-config per-participant plots ---
    _plot_best_config(all_rows, X_by_test, y_by, X_by, key, args)

    # --- Full JSON ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    print(f"JSON written to {args.out}")


# ===========================================================================
# CSV writers
# ===========================================================================

_ALL_PARAM_COLS = ["alpha", "window_n", "alpha_min", "alpha_max",
                   "lag_n", "process_noise", "measurement_noise"]


def _write_csv(all_rows: list[dict], args: argparse.Namespace) -> None:
    """Write per-participant CSV sorted by bal_acc descending."""
    import csv

    cols = ["pid", "seed", "cal_dur_s", "thresh_strategy",
            "config_id", "method",
            *_ALL_PARAM_COLS,
            "hysteresis", "threshold", "ft_auc",
            "bal_acc", "pct_on_hi", "pct_off_lo",
            "switch_rate_pm", "median_bout_s", "mean_onset_s"]

    sorted_rows = sorted(all_rows, key=lambda r: r.get("bal_acc", 0), reverse=True)

    _CSV_PID.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_PID, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in sorted_rows:
            out_row = {c: row.get(c, "") for c in cols}
            writer.writerow(out_row)

    print(f"Per-pid CSV written to {_CSV_PID}  ({len(sorted_rows):,} rows)")


def _write_summary(all_rows: list[dict], args: argparse.Namespace) -> None:
    """Write summary CSV: mean/std across pids and seeds, plus wins column."""
    import csv
    from collections import defaultdict

    # Group rows by (config_id, method, hysteresis, cal_dur_s, thresh_strategy)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["config_id"], r["method"], r["hysteresis"],
               r["cal_dur_s"], r["thresh_strategy"])
        groups[key].append(r)

    # Compute wins: for each pid, which config_id has highest bal_acc
    # at seed=0, cal_dur=30 for comparability
    ref_rows = [r for r in all_rows if r["seed"] == 0 and r["cal_dur_s"] == 30]
    wins_count: dict[str, int] = defaultdict(int)
    pid_groups: dict[str, list[dict]] = defaultdict(list)
    for r in ref_rows:
        pid_groups[r["pid"]].append(r)
    for pid, rrows in pid_groups.items():
        best = max(rrows, key=lambda r: r["bal_acc"])
        wins_count[best["config_id"]] += 1

    _METRIC_COLS = ["bal_acc", "pct_on_hi", "pct_off_lo",
                    "switch_rate_pm", "median_bout_s", "mean_onset_s"]

    summary_rows: list[dict] = []
    for (cfg_id, method, hyst, cal_dur, thresh_strat), rows in groups.items():
        # Grab param_dict from first row
        param_vals = {c: rows[0].get(c, "") for c in _ALL_PARAM_COLS}
        srow: dict = {
            "config_id": cfg_id,
            "method":    method,
            **param_vals,
            "thresh_strategy": thresh_strat,
            "hysteresis":  hyst,
            "cal_dur_s":   cal_dur,
            "n":           len(rows),
            "wins":        wins_count.get(cfg_id, 0),
        }
        for m in _METRIC_COLS:
            vals = [r[m] for r in rows if r[m] is not None and not (isinstance(r[m], float) and np.isnan(r[m]))]
            srow[f"mean_{m}"] = round(float(np.mean(vals)), 2) if vals else ""
            srow[f"std_{m}"]  = round(float(np.std(vals)), 2)  if vals else ""
        summary_rows.append(srow)

    summary_rows.sort(key=lambda r: r.get("mean_bal_acc", 0), reverse=True)

    cols = ["config_id", "method", *_ALL_PARAM_COLS,
            "thresh_strategy", "hysteresis", "cal_dur_s", "n", "wins"]
    for m in _METRIC_COLS:
        cols.extend([f"mean_{m}", f"std_{m}"])

    _CSV_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with open(_CSV_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for srow in summary_rows:
            writer.writerow({c: srow.get(c, "") for c in cols})

    print(f"Summary CSV written to {_CSV_SUMMARY}  ({len(summary_rows)} rows)")


# ===========================================================================
# Heatmap plots
# ===========================================================================

def _aggregate_for_plots(all_rows: list[dict]) -> dict[tuple, dict]:
    """Aggregate all_rows by (config_id, method, hysteresis) → mean metrics."""
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["config_id"], r["method"], r["hysteresis"])
        groups[key].append(r)

    agg: dict[tuple, dict] = {}
    for key, rows in groups.items():
        cfg_id, method, hyst = key
        param_vals = {c: rows[0].get(c, "") for c in _ALL_PARAM_COLS}
        vals_ba  = [r["bal_acc"] for r in rows if not (isinstance(r["bal_acc"], float) and np.isnan(r["bal_acc"]))]
        vals_sw  = [r["switch_rate_pm"] for r in rows if not (isinstance(r["switch_rate_pm"], float) and np.isnan(r["switch_rate_pm"]))]
        agg[key] = {
            **param_vals,
            "mean_bal_acc":       float(np.mean(vals_ba)) if vals_ba else float("nan"),
            "mean_switch_rate_pm": float(np.mean(vals_sw)) if vals_sw else float("nan"),
        }
    return agg


def _plot_heatmaps(all_rows: list[dict], args: argparse.Namespace) -> None:
    """Generate per-method heatmaps and hysteresis-effect line chart."""
    hm_dir = _FIG_DIR / "heatmaps"
    hm_dir.mkdir(parents=True, exist_ok=True)

    agg = _aggregate_for_plots(all_rows)

    # --- EMA bar chart ---
    ema_items = [(k, v) for k, v in agg.items() if k[1] == "ema" and k[2] == 0.0]
    if ema_items:
        ema_items.sort(key=lambda x: x[0][0])  # sort by config_id
        alphas   = [v["alpha"] for _, v in ema_items]
        bal_accs = [v["mean_bal_acc"] for _, v in ema_items]
        sw_rates = [v["mean_switch_rate_pm"] for _, v in ema_items]

        fig, ax = plt.subplots(figsize=(8, 4))
        norm = plt.Normalize(min(sw_rates), max(sw_rates))
        cmap = plt.cm.RdYlGn_r
        bars = ax.bar([str(a) for a in alphas], bal_accs,
                      color=[cmap(norm(s)) for s in sw_rates])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="mean switch_rate/min")
        ax.set_xlabel("alpha")
        ax.set_ylabel("mean bal_acc (%)")
        ax.set_title("EMA: bal_acc by alpha (hyst=0)")
        fig.tight_layout()
        fig.savefig(hm_dir / "ema_bar.png", dpi=150)
        plt.close(fig)

    # --- SMA bar chart ---
    sma_items = [(k, v) for k, v in agg.items() if k[1] == "sma" and k[2] == 0.0]
    if sma_items:
        sma_items.sort(key=lambda x: x[1]["window_n"])
        windows  = [v["window_n"] for _, v in sma_items]
        bal_accs = [v["mean_bal_acc"] for _, v in sma_items]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([str(w) for w in windows], bal_accs, color="steelblue")
        ax.set_xlabel("window_n")
        ax.set_ylabel("mean bal_acc (%)")
        ax.set_title("SMA: bal_acc by window size (hyst=0)")
        fig.tight_layout()
        fig.savefig(hm_dir / "sma_bar.png", dpi=150)
        plt.close(fig)

    # --- AdaptiveEMA 2D heatmaps (one per hysteresis) ---
    aema_items = [(k, v) for k, v in agg.items() if k[1] == "adaptive_ema"]
    if aema_items:
        hyst_vals = sorted(set(k[2] for k, _ in aema_items))
        for hyst in hyst_vals:
            sub = [(k, v) for k, v in aema_items if k[2] == hyst]
            amins = sorted(set(v["alpha_min"] for _, v in sub))
            amaxs = sorted(set(v["alpha_max"] for _, v in sub))
            grid = np.full((len(amins), len(amaxs)), float("nan"))
            for _, v in sub:
                ri = amins.index(v["alpha_min"])
                ci = amaxs.index(v["alpha_max"])
                grid[ri, ci] = v["mean_bal_acc"]

            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(amaxs)))
            ax.set_xticklabels([str(a) for a in amaxs])
            ax.set_yticks(range(len(amins)))
            ax.set_yticklabels([str(a) for a in amins])
            ax.set_xlabel("alpha_max")
            ax.set_ylabel("alpha_min")
            ax.set_title(f"AdaptiveEMA: mean bal_acc (hyst={hyst})")
            fig.colorbar(im, ax=ax, label="bal_acc (%)")
            fig.tight_layout()
            fig.savefig(hm_dir / f"aema_hyst{hyst}.png", dpi=150)
            plt.close(fig)

    # --- FixedLag 2D heatmaps (one per lag_n) ---
    fl_items = [(k, v) for k, v in agg.items() if k[1] == "fixed_lag" and k[2] == 0.0]
    if fl_items:
        lag_vals = sorted(set(v["lag_n"] for _, v in fl_items))
        for lag in lag_vals:
            sub = [(k, v) for k, v in fl_items if v["lag_n"] == lag]
            pns = sorted(set(v["process_noise"] for _, v in sub))
            mns = sorted(set(v["measurement_noise"] for _, v in sub))
            grid = np.full((len(pns), len(mns)), float("nan"))
            for _, v in sub:
                ri = pns.index(v["process_noise"])
                ci = mns.index(v["measurement_noise"])
                grid[ri, ci] = v["mean_bal_acc"]

            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(mns)))
            ax.set_xticklabels([str(m) for m in mns])
            ax.set_yticks(range(len(pns)))
            ax.set_yticklabels([str(p) for p in pns])
            ax.set_xlabel("measurement_noise")
            ax.set_ylabel("process_noise")
            ax.set_title(f"FixedLag (lag={lag}): mean bal_acc (hyst=0)")
            fig.colorbar(im, ax=ax, label="bal_acc (%)")
            fig.tight_layout()
            fig.savefig(hm_dir / f"fixedlag_l{lag}.png", dpi=150)
            plt.close(fig)

    # --- Hysteresis effect: line chart per method ---
    methods = sorted(set(k[1] for k in agg))
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        method_items = [(k, v) for k, v in agg.items() if k[1] == method]
        hyst_vals = sorted(set(k[2] for k, _ in method_items))
        mean_ba_by_hyst = []
        for h in hyst_vals:
            vals = [v["mean_bal_acc"] for k, v in method_items
                    if k[2] == h and not np.isnan(v["mean_bal_acc"])]
            mean_ba_by_hyst.append(float(np.mean(vals)) if vals else float("nan"))
        ax.plot(hyst_vals, mean_ba_by_hyst, marker="o", label=method)
    ax.set_xlabel("Hysteresis margin")
    ax.set_ylabel("mean bal_acc (%)")
    ax.set_title("Hysteresis effect by smoother method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(hm_dir / "hysteresis_effect.png", dpi=150)
    plt.close(fig)

    print(f"Heatmaps saved to {hm_dir}")


# ===========================================================================
# Top-N per-participant time-series plots
# ===========================================================================

def _plot_best_config(
    all_rows: list[dict],
    X_by_test: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    X_by: dict[str, np.ndarray],
    dataset_key: str,
    args: argparse.Namespace,
) -> None:
    """Re-simulate the single best config and save per-participant plots.

    Identifies the (thresh_strategy, config_id, hysteresis) combo with the
    highest mean bal_acc across all seeds, cal durations, and participants,
    then generates one 2-panel PNG per participant in a pid/ subdirectory
    (matching simulate_mwl_adaptation.py layout).
    """
    from collections import defaultdict

    # Aggregate mean_bal_acc by (thresh_strategy, config_id, hysteresis)
    groups: dict[tuple, list[float]] = defaultdict(list)
    cfg_lookup: dict[str, tuple[MwlSmootherConfig, dict]] = {}
    thresh_lookup: dict[tuple, float] = {}  # store a representative threshold
    for r in all_rows:
        key = (r["thresh_strategy"], r["config_id"], r["hysteresis"])
        groups[key].append(r["bal_acc"])
        if r["config_id"] not in cfg_lookup:
            cfg_lookup[r["config_id"]] = _cfg_from_row(r)

    ranked = sorted(groups.items(),
                    key=lambda x: float(np.nanmean(x[1])), reverse=True)
    (best_thresh_strat, best_cfg_id, best_hyst), best_vals = ranked[0]
    best_ba = float(np.nanmean(best_vals))
    smoother_cfg, _ = cfg_lookup[best_cfg_id]

    print(f"\nBest config: {best_cfg_id}  thresh={best_thresh_strat}  "
          f"hyst={best_hyst}  mean_bal_acc={best_ba:.1f}%")

    pids = sorted(X_by_test.keys())
    if args.only:
        pids = [p for p in pids if p in args.only]

    out_dir = _FIG_DIR / "best_config"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in pids:
        selector, group_pipe, _ = _load_or_train_group_model(
            X_by, y_by, pid, dataset_key)
        sc  = group_pipe.named_steps["sc"]
        nys = group_pipe.named_steps["nys"]

        X_full = X_by_test[pid]
        y_full = y_by[pid]

        # Cal split: seed=0, cal_dur=60 (first retained cal duration)
        pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(SEED + pid_hash + 0 * 1000)
        X_cal, y_cal, _, _, _ = _random_cal_split(
            X_full, y_full, _CAL_DURATIONS[0], rng)

        clf, probs_cal = _fit_fthead(selector, group_pipe, X_cal, y_cal)

        # Derive threshold using the best strategy
        thresh_list = _compute_thresholds(y_cal, probs_cal)
        threshold = dict(thresh_list)[best_thresh_strat]

        # Full-session probabilities
        X_sel = selector.transform(X_full)
        X_proj = nys.transform(sc.transform(X_sel))
        p_high_all = clf.predict_proba(X_proj)[:, 1]

        # Simulate with hysteresis
        smoother = _make_smoother(smoother_cfg)
        n_epochs = len(y_full)
        smoothed_arr = np.zeros(n_epochs)
        assist_on    = np.zeros(n_epochs, dtype=bool)
        state_on = False
        for i in range(n_epochs):
            sm = smoother.update(float(p_high_all[i]))
            smoothed_arr[i] = sm
            if state_on:
                if sm < threshold - best_hyst:
                    state_on = False
            else:
                if sm >= threshold + best_hyst:
                    state_on = True
            assist_on[i] = state_on

        # Per-participant directory (matches simulate_mwl_adaptation.py)
        pid_dir = out_dir / pid
        pid_dir.mkdir(parents=True, exist_ok=True)

        _plot_two_panel(
            pid, y_full, p_high_all, smoothed_arr,
            assist_on, threshold,
            f"{best_cfg_id}  {best_thresh_strat}  hyst={best_hyst}",
            pid_dir / "best_smoother.png",
        )

    print(f"Best-config plots saved to {out_dir}")


def _cfg_from_row(row: dict) -> tuple[MwlSmootherConfig, dict]:
    """Reconstruct MwlSmootherConfig and param_dict from a result row."""
    method = row["method"]
    if method == "ema":
        cfg = MwlSmootherConfig(method="ema", alpha=row["alpha"])
        return cfg, {"alpha": row["alpha"]}
    if method == "sma":
        cfg = MwlSmootherConfig(method="sma", window_n=row["window_n"])
        return cfg, {"window_n": row["window_n"]}
    if method == "adaptive_ema":
        cfg = MwlSmootherConfig(method="adaptive_ema",
                                alpha_min=row["alpha_min"],
                                alpha_max=row["alpha_max"])
        return cfg, {"alpha_min": row["alpha_min"], "alpha_max": row["alpha_max"]}
    if method == "fixed_lag":
        cfg = MwlSmootherConfig(method="fixed_lag",
                                lag_n=row["lag_n"],
                                process_noise=row["process_noise"],
                                measurement_noise=row["measurement_noise"])
        return cfg, {"lag_n": row["lag_n"],
                     "process_noise": row["process_noise"],
                     "measurement_noise": row["measurement_noise"]}
    raise ValueError(f"Unknown method: {method}")


def _plot_two_panel(
    pid: str,
    y_full: np.ndarray,
    p_high_all: np.ndarray,
    smoothed_all: np.ndarray,
    assist_all: np.ndarray,
    threshold: float,
    title_suffix: str,
    out_path: Path,
) -> None:
    """Two-panel plot (LOW MWL | HIGH MWL) showing when assistance is ON/OFF."""
    lo_idx = np.where(y_full == 0)[0]
    hi_idx = np.where(y_full == 1)[0]
    panels = [
        ("LOW MWL",  lo_idx, "tab:blue"),
        ("HIGH MWL", hi_idx, "tab:orange"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for pi, (panel_name, idx, bg_color) in enumerate(panels):
        ax = axes[pi]
        t_panel  = np.arange(len(idx)) * _STEP_S
        p_panel  = p_high_all[idx]
        sm_panel = smoothed_all[idx]
        on_panel = assist_all[idx]

        ax.axvspan(t_panel[0], t_panel[-1], color=bg_color, alpha=0.08)

        for j in range(len(t_panel)):
            if on_panel[j]:
                x0 = t_panel[j] - _STEP_S / 2
                x1 = t_panel[j] + _STEP_S / 2
                ax.axvspan(max(x0, t_panel[0]), min(x1, t_panel[-1]),
                           color="green", alpha=0.07)

        ax.plot(t_panel, p_panel, color="0.65", alpha=0.5, linewidth=0.5,
                label="raw" if pi == 0 else None)
        ax.plot(t_panel, sm_panel, color="0.2", linewidth=1.0,
                label="smoothed" if pi == 0 else None)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8)

        pct_on = 100.0 * on_panel.mean()
        ax.set_title(f"{panel_name}  —  assist ON {pct_on:.0f}%", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        if pi == 0:
            ax.set_ylabel("P(high MWL)")
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"{pid}  —  {title_suffix}   thresh={threshold:.3f}   green = assist ON",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
