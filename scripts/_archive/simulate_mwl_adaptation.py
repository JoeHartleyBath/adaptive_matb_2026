"""simulate_mwl_adaptation.py

Replay real EEG-derived MWL estimates to test when assistance would switch
ON or OFF based on the smoothed workload probability and Youden-J threshold.

For each of the 28 included participants, the simulation:
  1. Trains a group RBF model on the other 27 participants (LOSO).
  2. Randomly samples 30 s of calibration data per label to fit a personalised
     FT-head classifier (LogReg on Nystroem-projected cal data).
  3. Replays all 4 MATB condition blocks in chronological order, computing
     smoothed P(high_MWL) and comparing to the Youden-J threshold.
  4. Records the binary assistance state (ON/OFF) for three smoothers:
     EMA(α=0.10), SMA(window=8), AdaptiveEMA.

Assistance logic: assistance is ON when smoothed P(high_MWL) >= threshold,
OFF otherwise.  In HIGH-MWL blocks assistance should be ON; in LOW-MWL
blocks it should be OFF.

Caveats:
  - Oracle calibration: cal data drawn randomly from all 4 blocks (same as
    personalisation_comparison.py).  Real deployment would accumulate cal data
    sequentially, so AUC is slightly optimistic here.
  - No timing jitter: 4 blocks replayed contiguously; inter-block Forest
    relaxation periods (~180 s) are not simulated.
  - Low-AUC participants: P01, P17, P19, P30 have FT-head AUC ≤ 0.45 even
    with personalisation; their d(t) trajectories will be noisy.

Usage:
    python scripts/simulate_mwl_adaptation.py
    python scripts/simulate_mwl_adaptation.py --only P05
    python scripts/simulate_mwl_adaptation.py --out results/test_pretrain/custom.json
"""
from __future__ import annotations

import argparse
import hashlib
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
_DEFAULT_OUT   = _REPO_ROOT / "results" / "test_pretrain" / "simulate_mwl_adaptation.json"
_FIG_DIR       = _REPO_ROOT / "results" / "figures" / "mwl_adaptation"
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


# ===========================================================================
# FT-head personalisation
# ===========================================================================

def _fit_fthead(
    selector: SelectKBest,
    group_pipe: Pipeline,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[LogisticRegression, float]:
    """Fit a personalised LogReg head on Nystroem-projected calibration data.

    Projects X_cal through selector → StandardScaler → Nystroem (from the
    group pipeline), then fits a new LogisticRegression on the projected
    features.  Computes a Youden-J threshold from the cal predictions.

    Returns:
        (fitted_clf, threshold)
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
    threshold, _ = _youden_threshold(y_cal, probs_cal)

    return clf, threshold


# ===========================================================================
# Per-smoother simulation
# ===========================================================================

_SMOOTHER_CFGS = {
    "ema": MwlSmootherConfig(method="ema", alpha=0.10),
    "sma": MwlSmootherConfig(method="sma", window_n=8),
    "adaptive_ema": MwlSmootherConfig(method="adaptive_ema"),
    "fixed_lag": MwlSmootherConfig(method="fixed_lag", lag_n=3,
                                    process_noise=0.005, measurement_noise=0.1),
}


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


def _simulate_one_smoother(
    pid: str,
    X_full: np.ndarray,
    y_full: np.ndarray,
    selector: SelectKBest,
    group_pipe: Pipeline,
    clf_personal: LogisticRegression,
    threshold: float,
    smoother_cfg: MwlSmootherConfig,
) -> dict:
    """Simulate assistance switching for one participant with one smoother.

    Replays all epochs in chronological order, computes smoothed P(high_MWL),
    and records whether assistance would be ON (smoothed >= threshold) or OFF.

    Returns a dict with per-epoch arrays.
    """
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]

    smoother = _make_smoother(smoother_cfg)

    n_epochs = len(X_full)
    ts           = np.zeros(n_epochs)
    p_highs      = np.zeros(n_epochs)
    smoothed_arr = np.zeros(n_epochs)
    assist_on    = np.zeros(n_epochs, dtype=bool)

    for i in range(n_epochs):
        t = i * _STEP_S

        # Project epoch through group pipeline feature path
        X_sel = selector.transform(X_full[i : i + 1])
        X_proj = nys.transform(sc.transform(X_sel))
        p_high = float(clf_personal.predict_proba(X_proj)[0, 1])

        smoothed = smoother.update(p_high)

        ts[i]           = t
        p_highs[i]      = p_high
        smoothed_arr[i] = smoothed
        assist_on[i]    = smoothed >= threshold

    return {
        "t":        ts.tolist(),
        "p_high":   p_highs.tolist(),
        "smoothed": smoothed_arr.tolist(),
        "assist_on": assist_on.tolist(),
    }


# ===========================================================================
# Per-smoother quality metrics
# ===========================================================================

def _compute_smoother_stats(
    assist_on: np.ndarray,
    y_full: np.ndarray,
) -> dict:
    """Compute quality metrics for one smoother's assist_on predictions.

    Parameters
    ----------
    assist_on : bool array, one per epoch
    y_full : int array, actual MWL labels (0=LOW, 1=HIGH)

    Returns dict with:
        pct_on_hi        % of HIGH epochs where assist is ON
        pct_off_lo       % of LOW epochs where assist is OFF
        bal_acc          balanced accuracy treating assist_on as classifier
        switch_rate_pm   ON<->OFF transitions per minute
        median_bout_s    median duration of contiguous ON or OFF bouts (s)
        onset_latencies  list of seconds until first ON in each HIGH block
    """
    blocks = _detect_blocks(y_full)

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
# Per-participant plot
# ===========================================================================

def _plot_participant(
    pid: str,
    y_full: np.ndarray,
    results_by_smoother: dict[str, dict],
    threshold: float,
    out_dir: Path,
) -> None:
    """Two-panel plot (LOW MWL | HIGH MWL) showing when assistance is ON/OFF.

    Generates one PNG per smoother.  Each PNG has two panels: LEFT = all
    LOW-MWL epochs concatenated, RIGHT = all HIGH-MWL epochs concatenated.
    Within each panel the epoch-local time axis is continuous.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_dir = out_dir / pid
    pid_dir.mkdir(parents=True, exist_ok=True)

    # Gather epoch indices for each condition
    lo_idx = np.where(y_full == 0)[0]
    hi_idx = np.where(y_full == 1)[0]
    panels = [
        ("LOW MWL",  lo_idx, "tab:blue"),
        ("HIGH MWL", hi_idx, "tab:orange"),
    ]

    smoother_labels = {
        "ema":          "EMA(\u03b1=0.10)",
        "sma":          "SMA(w=8)",
        "adaptive_ema": "AdaptiveEMA",
        "fixed_lag":    "FixedLag(L=3)",
    }

    for sname, slabel in smoother_labels.items():
        res = results_by_smoother[sname]
        p_high_all   = np.array(res["p_high"])
        smoothed_all = np.array(res["smoothed"])
        assist_all   = np.array(res["assist_on"])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for pi, (panel_name, idx, bg_color) in enumerate(panels):
            ax = axes[pi]
            t_panel  = np.arange(len(idx)) * _STEP_S
            p_panel  = p_high_all[idx]
            sm_panel = smoothed_all[idx]
            on_panel = np.array(assist_all)[idx]

            # Background shading for condition
            ax.axvspan(t_panel[0], t_panel[-1], color=bg_color, alpha=0.08)

            # Assistance ON shading (subtle green)
            for j in range(len(t_panel)):
                if on_panel[j]:
                    x0 = t_panel[j] - _STEP_S / 2
                    x1 = t_panel[j] + _STEP_S / 2
                    ax.axvspan(max(x0, t_panel[0]), min(x1, t_panel[-1]),
                               color="green", alpha=0.07)

            # Raw P(high MWL) trace
            ax.plot(t_panel, p_panel, color="0.65", alpha=0.5, linewidth=0.5,
                    label="raw" if pi == 0 else None)
            # Smoothed trace
            ax.plot(t_panel, sm_panel, color="0.2", linewidth=1.0,
                    label="smoothed" if pi == 0 else None)

            # Threshold
            ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8)

            pct_on = 100.0 * on_panel.mean()
            ax.set_title(f"{panel_name}  —  assist ON {pct_on:.0f}%",
                         fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            if pi == 0:
                ax.set_ylabel("P(high MWL)")
                ax.legend(fontsize=7, loc="lower right")

        fig.suptitle(
            f"{pid}  \u2014  {slabel}   "
            f"thresh={threshold:.3f}   "
            f"green = assist ON",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(pid_dir / f"{sname}.png", dpi=150)
        plt.close(fig)


# ===========================================================================
# Per-participant pipeline (parallelisable)
# ===========================================================================

def _process_one_participant(
    pid: str,
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    X_by_test: dict[str, np.ndarray],
    dataset_key: str,
    smoother_cfgs: dict[str, MwlSmootherConfig],
) -> tuple[str, dict, float]:
    """Full simulation pipeline for one participant (no plotting).

    Returns (pid, results_dict, elapsed_seconds).
    """
    t0 = time.time()

    # 1. Train group model (LOSO) — cached across runs
    selector, group_pipe, group_info = _load_or_train_group_model(
        X_by, y_by, pid, dataset_key)

    # 2. Cal split (30 s per label, deterministic rng)
    pid_hash = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.default_rng(SEED + pid_hash)
    X_cal, y_cal, X_test, y_test, split_info = _random_cal_split(
        X_by_test[pid], y_by[pid], _CAL_DURATION_S, rng)

    # 3. Fit FT-head + Youden-J threshold
    clf_personal, threshold = _fit_fthead(
        selector, group_pipe, X_cal, y_cal)

    # 4. FT-head AUC on test portion
    X_test_sel = selector.transform(X_test)
    sc  = group_pipe.named_steps["sc"]
    nys = group_pipe.named_steps["nys"]
    probs_test = clf_personal.predict_proba(
        nys.transform(sc.transform(X_test_sel)))[:, 1]
    ft_auc = _auc(y_test, probs_test)

    # 5. Simulate all smoothers
    X_full = X_by_test[pid]
    y_full = y_by[pid]
    pid_results: dict[str, dict] = {}
    for sname, scfg in smoother_cfgs.items():
        pid_results[sname] = _simulate_one_smoother(
            pid, X_full, y_full,
            selector, group_pipe, clf_personal, threshold, scfg)

    pid_results["ft_auc"]     = ft_auc
    pid_results["threshold"]  = threshold
    pid_results["group_info"] = group_info
    pid_results["split_info"] = split_info

    # 6. Per-smoother quality stats
    y_full_arr = y_by[pid]
    for sname in smoother_cfgs:
        on_arr = np.array(pid_results[sname]["assist_on"])
        pid_results[sname]["stats"] = _compute_smoother_stats(on_arr, y_full_arr)

    return pid, pid_results, time.time() - t0


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
# main
# ===========================================================================

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Simulate MWL-driven adaptation with StaircaseController")
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Run only these participant IDs (e.g., --only P05)")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all cores, 1 = sequential)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached group models before running")
    args = parser.parse_args()

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

    # Per-participant normalisation (mixed: causal for test)
    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    X_by_test: dict[str, np.ndarray] | None = None
    if baseline_by is not None:
        X_by, X_by_test = prepare_mixed_norm(X_by, baseline_by)
        print("Mixed normalisation: pp z-score (train) + calibration (test).")
    else:
        for pid in X_by:
            sc_pp = StandardScaler()
            X_by[pid] = sc_pp.fit_transform(X_by[pid])
        print("Per-participant z-scoring applied (baseline cache not found).")

    # Filter to --only if specified
    if args.only:
        pids = [p for p in pids if p in args.only]
        if not pids:
            print(f"ERROR: None of {args.only} found in included participants.")
            sys.exit(1)

    print(f"Loaded {len(pids)} participants, {len(feat_names)} features  [key={key}]")

    # Use calibration-normed features for held-out; fall back to X_by
    if X_by_test is None:
        X_by_test = X_by

    # ------------------------------------------------------------------
    # Clear group model cache if requested
    # ------------------------------------------------------------------
    if args.clear_cache and _GROUP_CACHE.exists():
        import shutil
        shutil.rmtree(_GROUP_CACHE)
        print("Cleared group model cache.")

    # ------------------------------------------------------------------
    # Participant loop (parallel, with cached group models)
    # ------------------------------------------------------------------
    print(f"\nProcessing {len(pids)} participants ({args.jobs} jobs)...\n")
    t_start = time.time()

    raw_results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(_process_one_participant)(
            pid, X_by, y_by, X_by_test, key, _SMOOTHER_CFGS,
        )
        for pid in pids
    )

    t_total = (time.time() - t_start) / 60

    # -- Summary table -------------------------------------------------
    results: dict[str, dict] = {}
    snames = list(_SMOOTHER_CFGS.keys())
    hdr_parts = [f"{'PID':>5}", f"{'AUC':>6}", f"{'Thr':>5}"]
    for sn in snames:
        hdr_parts.append(f"{'BA%':>5}")
        hdr_parts.append(f"{'sw/m':>5}")
    print("  ".join(hdr_parts))
    sub_parts = [" " * 5, " " * 6, " " * 5]
    for sn in snames:
        label = sn[:10].center(12)
        sub_parts.append(label)
    print("  ".join(sub_parts))
    print("-" * (22 + 14 * len(snames)))

    for pid, pid_results, elapsed in sorted(raw_results, key=lambda x: x[0]):
        results[pid] = pid_results
        ft_auc    = pid_results["ft_auc"]
        threshold = pid_results["threshold"]
        parts = [f"{pid:>5}", f"{ft_auc:>6.3f}", f"{threshold:>5.3f}"]
        for sname in snames:
            st = pid_results[sname]["stats"]
            parts.append(f"{st['bal_acc']:>5.1f}")
            parts.append(f"{st['switch_rate_pm']:>5.1f}")
        print("  ".join(parts) + f"  ({elapsed:.1f}s)")

    print(f"\nDone. {len(pids)} participants in {t_total:.1f} min.")

    # -- Plots (main process only) -------------------------------------
    print("Generating plots...")
    for pid in sorted(results):
        pr = results[pid]
        y_full = y_by[pid]
        _plot_participant(pid, y_full, pr, pr["threshold"], _FIG_DIR)
    print(f"Plots saved to {_FIG_DIR}")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    out_dict: dict = {
        "config": {
            "cal_duration_s": _CAL_DURATION_S,
            "step_s": _STEP_S,
            "gap_radius": _GAP_RADIUS,
            "seed": SEED,
            "smoothers": {
                sn: {"method": cfg.method, "alpha": cfg.alpha,
                     "window_n": cfg.window_n, "lag_n": cfg.lag_n,
                     "process_noise": cfg.process_noise,
                     "measurement_noise": cfg.measurement_noise}
                for sn, cfg in _SMOOTHER_CFGS.items()
            },
        },
        "participants": {},
    }
    for pid, pr in results.items():
        p_out: dict = {
            "ft_auc": round(pr["ft_auc"], 4),
            "threshold": round(pr["threshold"], 4),
            "group_info": pr["group_info"],
            "split_info": pr["split_info"],
        }
        for sn in _SMOOTHER_CFGS:
            p_out[sn] = {
                "smoothed": pr[sn]["smoothed"],
                "assist_on": pr[sn]["assist_on"],
                "stats": pr[sn]["stats"],
            }
        out_dict["participants"][pid] = p_out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_dict, indent=1), encoding="utf-8")
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
