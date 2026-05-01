"""sweep_mwl_smoothers_logreg.py

Sweep smoother hyperparameters, calibration durations, and hysteresis margins
across all included participants using LOSO — using the validated LogReg
WS-weak (C=0.1) personalisation pipeline from dc_logreg_personalisation.

Group model:  LogReg K=30, C=0.001, L2, StandardScaler (frozen, no inner CV).
Personalisation:  warm-start from group weights with C=0.1.
Normalisation:  calibration norm (fixation + Forest_0 baseline) for ALL pids.

Usage:
    python scripts/sweep_mwl_smoothers_logreg.py
    python scripts/sweep_mwl_smoothers_logreg.py --only P05 --jobs 1
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
import yaml
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
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
from ml.pretrain_loader import (                                         # noqa: E402
    calibration_norm_features,
    load_baseline_from_cache,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_NORM_CACHE    = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_QC_CONFIG     = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_DEFAULT_OUT   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep_logreg.json"
_CSV_PID       = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep_logreg.csv"
_CSV_SUMMARY   = _REPO_ROOT / "results" / "test_pretrain" / "smoother_sweep_logreg_summary.csv"
_FIG_DIR       = _REPO_ROOT / "results" / "figures" / "smoother_sweep_logreg"
_GROUP_CACHE   = _REPO_ROOT / "results" / "test_pretrain" / "group_logreg_cache"

# Frozen LogReg config (from dc_logreg_hyperparameter_plateau)
_LOGREG_K = 30
_LOGREG_C = 0.001

# WS-weak personalisation C (from dc_logreg_personalisation_comparison)
_WARM_C_WEAK = 0.1

# Simulation parameters
_STEP_S      = 0.5    # epoch step in seconds (matches HDF5 export)
_GAP_RADIUS  = 3      # epochs excluded around each cal chunk boundary
SEED = 42


def _load_exclude(cfg_path: Path) -> set[str]:
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_EXCLUDE = _load_exclude(_QC_CONFIG)


# ===========================================================================
# Pipeline factory
# ===========================================================================

def _make_logreg(C: float = _LOGREG_C, seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000,
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
    """Find the optimal classification threshold via Youden's J statistic."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(j[best_idx])


# ===========================================================================
# Group model training (LOSO — frozen LogReg, no inner CV)
# ===========================================================================

def _train_group_logreg(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    held_out_pid: str,
) -> tuple[SelectKBest, Pipeline, dict]:
    """Train frozen LogReg on all participants except held_out_pid.

    Config from dc_logreg_hyperparameter_plateau: K=30, C=0.001, L2,
    StandardScaler.  No inner CV needed (plateau confirmed).
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


def _load_or_train_group_logreg(
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
    selector, pipe, info = _train_group_logreg(X_by, y_by, held_out_pid)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((selector, pipe, info), cache_path, compress=3)
    return selector, pipe, info


# ===========================================================================
# Block detection & random calibration split
# ===========================================================================

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

    For each condition block, selects a contiguous chunk of the required size
    starting at a random offset.  A gap of _GAP_RADIUS epochs on each side
    of the chunk is excluded from the test set to prevent window-overlap
    leakage.
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
# WS-weak personalisation
# ===========================================================================

def _fit_ws_weak(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray]:
    """Fit a personalised WS-weak LogReg head on calibration data.

    Copies the group model's weights as warm-start initialisation, then
    refits with C=0.1 (weaker regularisation → more individual adaptation).

    Returns:
        (fitted_clf, probs_cal) — cal-set predicted probabilities for
        downstream threshold computation.
    """
    sc        = group_pipe.named_steps["sc"]
    group_clf = group_pipe.named_steps["clf"]

    X_cal_sel = selector.transform(X_cal)
    X_cal_sc  = sc.transform(X_cal_sel)

    clf = LogisticRegression(C=_WARM_C_WEAK, max_iter=2000,
                             warm_start=True, class_weight="balanced",
                             random_state=SEED)
    clf.classes_   = group_clf.classes_.copy()
    clf.coef_      = group_clf.coef_.copy()
    clf.intercept_ = group_clf.intercept_.copy()
    clf.fit(X_cal_sc, y_cal)

    probs_cal = clf.predict_proba(X_cal_sc)[:, 1]
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
        When provided, accuracy metrics are computed only on these epochs
        to avoid cal-data leakage.  Switching and bout metrics are always
        computed on the full trace (they reflect deployment behaviour).
    """
    blocks = _detect_blocks(y_full)

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

    sens = assist_on[hi_epochs].mean() if len(hi_epochs) else 0.0
    spec = (~assist_on[lo_epochs]).mean() if len(lo_epochs) else 0.0
    bal_acc = 100.0 * (sens + spec) / 2.0

    transitions = int(np.sum(np.diff(assist_on.astype(int)) != 0))
    total_time_min = len(assist_on) * _STEP_S / 60.0
    switch_rate_pm = transitions / total_time_min if total_time_min > 0 else 0.0

    bout_lengths: list[float] = []
    if len(assist_on) > 0:
        run_start = 0
        for i in range(1, len(assist_on)):
            if assist_on[i] != assist_on[run_start]:
                bout_lengths.append((i - run_start) * _STEP_S)
                run_start = i
        bout_lengths.append((len(assist_on) - run_start) * _STEP_S)
    median_bout_s = float(np.median(bout_lengths)) if bout_lengths else 0.0

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
# Data loading (calibration norm for all participants)
# ===========================================================================

def _load_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
                           list[str], list[str]]:
    """Load calibration-normalised features from the norm cache.

    Returns (X_by, y_by, feat_names, pids).
    All participants receive causal calibration normalisation (ADR-0004).
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
    for pid in available:
        X_by_raw[pid] = npz[f"{pid}_task_X"]
        y_by[pid] = npz[f"{pid}_task_y"]

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
    return X_by, y_by, feat_names, pids


def _dataset_key() -> str:
    """Derive a short cache key from the norm cache file mtime."""
    mtime = str(_NORM_CACHE.stat().st_mtime) if _NORM_CACHE.exists() else "missing"
    excl  = str(sorted(_EXCLUDE))
    raw   = "|".join([mtime, excl, "logreg_ws_weak"])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


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
    """Simulate assistance switching with stateful hysteresis."""
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

_CAL_DURATIONS = [60, 120]        # seconds per label
_SEEDS         = [0, 1, 2, 3, 4]
_HYST_MARGINS  = [0.00, 0.02, 0.05, 0.08]

_YOUDEN_OFFSETS   = [-0.05, 0.00, +0.05]
_FIXED_THRESHOLDS = [0.50, 0.60]
_COST_WEIGHTS     = [0.6, 0.7]


def _compute_thresholds(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
) -> list[tuple[str, float]]:
    """Return list of (strategy_id, threshold) for all 7 strategies."""
    base_thresh, _ = _youden_threshold(y_cal, probs_cal)

    results: list[tuple[str, float]] = []

    for off in _YOUDEN_OFFSETS:
        t = np.clip(base_thresh + off, 0.01, 0.99)
        label = f"youden{off:+.2f}" if off != 0.0 else "youden"
        results.append((label, float(t)))

    for t in _FIXED_THRESHOLDS:
        results.append((f"fixed_{t:.2f}", t))

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

    for alpha in [0.03, 0.05, 0.08, 0.10]:
        cfg = MwlSmootherConfig(method="ema", alpha=alpha)
        configs.append((f"ema_a{alpha}", cfg, {"alpha": alpha}))

    for w in [10, 12, 16]:
        cfg = MwlSmootherConfig(method="sma", window_n=w)
        configs.append((f"sma_w{w}", cfg, {"window_n": w}))

    for amin, amax in itertools.product(
        [0.03, 0.05], [0.15, 0.20, 0.30]
    ):
        if amin >= amax:
            continue
        cfg = MwlSmootherConfig(method="adaptive_ema", alpha_min=amin, alpha_max=amax)
        configs.append((f"aema_{amin}_{amax}", cfg, {"alpha_min": amin, "alpha_max": amax}))

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
    dataset_key: str,
    smoother_grid: list[tuple[str, MwlSmootherConfig, dict]],
) -> list[dict]:
    """Returns list of flat result dicts — one per (seed, cal_dur, smoother, hysteresis)."""
    selector, group_pipe, _ = _load_or_train_group_logreg(
        X_by, y_by, pid, dataset_key)
    sc = group_pipe.named_steps["sc"]

    X_full = X_by[pid]
    y_full = y_by[pid]

    # Pre-project all epochs through selector + scaler (do once)
    X_sel    = selector.transform(X_full)
    X_sc_all = sc.transform(X_sel)          # (N, K)

    rows: list[dict] = []
    pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)

    for seed in _SEEDS:
        for cal_dur in _CAL_DURATIONS:
            rng = np.random.default_rng(SEED + pid_hash + seed * 1000)
            X_cal, y_cal, X_test, y_test, split_info = _random_cal_split(
                X_full, y_full, cal_dur, rng)

            clf, probs_cal = _fit_ws_weak(selector, group_pipe, X_cal, y_cal)

            # WS-weak AUC on test portion
            X_test_sel = selector.transform(X_test)
            X_test_sc  = sc.transform(X_test_sel)
            probs_test = clf.predict_proba(X_test_sc)[:, 1]
            ws_auc     = _auc(y_test, probs_test)

            # Full-session probabilities for smoother simulation
            p_high_all = clf.predict_proba(X_sc_all)[:, 1]

            test_indices = _get_test_indices(
                y_full, cal_dur, SEED + pid_hash + seed * 1000)

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
                            "ws_auc":        round(ws_auc, 4),
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
        description="Sweep smoother hyperparameters (LogReg WS-weak pipeline)")
    parser.add_argument("--out",     type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only",    type=str,  nargs="*", default=None)
    parser.add_argument("--jobs",    type=int,  default=4)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load features (calibration-normalised for all participants)
    # ------------------------------------------------------------------
    X_by, y_by, _, pids = _load_data()
    if args.only:
        pids = [p for p in pids if p in args.only]

    key = _dataset_key()

    smoother_grid = _build_smoother_grid()
    n_thresh = len(_YOUDEN_OFFSETS) + len(_FIXED_THRESHOLDS) + len(_COST_WEIGHTS)
    total = (len(pids) * len(_SEEDS) * len(_CAL_DURATIONS)
             * n_thresh * len(smoother_grid) * len(_HYST_MARGINS))
    print(f"LogReg WS-weak smoother sweep")
    print(f"  Group: K={_LOGREG_K}, C={_LOGREG_C}")
    print(f"  WS-weak C={_WARM_C_WEAK}")
    print(f"  Norm: calibration (fixation + Forest_0)")
    print(f"  {len(pids)} pids × {len(_SEEDS)} seeds × {len(_CAL_DURATIONS)} cal × "
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
                pid, X_by, y_by, key, smoother_grid
            ) for pid in batch_pids
        )
        for rows in batch_results:
            all_rows.extend(rows)
        done_pids.extend(batch_pids)

        elapsed = (time.time() - t0) / 60
        _print_running_best(all_rows, done_pids, elapsed, len(pids))

    print(f"Done in {(time.time()-t0)/60:.1f} min  — {len(all_rows):,} rows")

    # --- CSV output ---
    _write_csv(all_rows, args)

    # --- Summary CSV ---
    _write_summary(all_rows, args)

    # --- Heatmaps ---
    _plot_heatmaps(all_rows, args)

    # --- Best-config per-participant plots ---
    _plot_best_config(all_rows, X_by, y_by, key, args)

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
            "hysteresis", "threshold", "ws_auc",
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

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["config_id"], r["method"], r["hysteresis"],
               r["cal_dur_s"], r["thresh_strategy"])
        groups[key].append(r)

    # Wins: for each pid at seed=0, cal_dur=60, which config_id has highest bal_acc
    ref_rows = [r for r in all_rows if r["seed"] == 0 and r["cal_dur_s"] == 60]
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
        ema_items.sort(key=lambda x: x[0][0])
        alphas   = [v["alpha"] for _, v in ema_items]
        bal_accs = [v["mean_bal_acc"] for _, v in ema_items]
        sw_rates = [v["mean_switch_rate_pm"] for _, v in ema_items]

        fig, ax = plt.subplots(figsize=(8, 4))
        norm = plt.Normalize(min(sw_rates), max(sw_rates))
        cmap = plt.cm.RdYlGn_r
        ax.bar([str(a) for a in alphas], bal_accs,
               color=[cmap(norm(s)) for s in sw_rates])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="mean switch_rate/min")
        ax.set_xlabel("alpha")
        ax.set_ylabel("mean bal_acc (%)")
        ax.set_title("EMA: bal_acc by alpha (hyst=0) — LogReg WS-weak")
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
        ax.set_title("SMA: bal_acc by window size (hyst=0) — LogReg WS-weak")
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
            ax.set_title(f"AdaptiveEMA: mean bal_acc (hyst={hyst}) — LogReg WS-weak")
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
            ax.set_title(f"FixedLag (lag={lag}): mean bal_acc (hyst=0) — LogReg WS-weak")
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
    ax.set_title("Hysteresis effect by smoother method — LogReg WS-weak")
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
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    dataset_key: str,
    args: argparse.Namespace,
) -> None:
    """Re-simulate the single best config and save per-participant plots."""
    from collections import defaultdict

    groups: dict[tuple, list[float]] = defaultdict(list)
    cfg_lookup: dict[str, tuple[MwlSmootherConfig, dict]] = {}
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

    pids = sorted(X_by.keys())
    if args.only:
        pids = [p for p in pids if p in args.only]

    out_dir = _FIG_DIR / "best_config"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in pids:
        selector, group_pipe, _ = _load_or_train_group_logreg(
            X_by, y_by, pid, dataset_key)
        sc = group_pipe.named_steps["sc"]

        X_full = X_by[pid]
        y_full = y_by[pid]

        # Cal split: seed=0, first cal duration
        pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.default_rng(SEED + pid_hash + 0 * 1000)
        X_cal, y_cal, _, _, _ = _random_cal_split(
            X_full, y_full, _CAL_DURATIONS[0], rng)

        clf, probs_cal = _fit_ws_weak(selector, group_pipe, X_cal, y_cal)

        # Derive threshold using the best strategy
        thresh_list = _compute_thresholds(y_cal, probs_cal)
        threshold = dict(thresh_list)[best_thresh_strat]

        # Full-session probabilities
        X_sel  = selector.transform(X_full)
        X_sc   = sc.transform(X_sel)
        p_high_all = clf.predict_proba(X_sc)[:, 1]

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

        pid_dir = out_dir / pid
        pid_dir.mkdir(parents=True, exist_ok=True)

        cal_dur = _CAL_DURATIONS[0]
        _plot_two_panel(
            pid, y_full, p_high_all, smoothed_arr,
            assist_on, threshold,
            f"{best_cfg_id}  {best_thresh_strat}  hyst={best_hyst}  cal={cal_dur}s",
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
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
