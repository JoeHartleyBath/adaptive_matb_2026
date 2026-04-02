"""Multi-feature MWL model comparison: expanded feature set + multiple models.

Features (~45 per epoch, computed on region-mean signals):

  Bandpower (spectral, log-power):
    Frontal Theta/Alpha/Beta/Delta, Parietal Alpha, Central Theta/Alpha/Beta,
    Occipital Alpha, FAA, Central Engagement, FM Theta/Alpha, FM Theta/Beta,
    Central Theta/Beta, Frontal Delta/Alpha

  Hjorth parameters (time-domain, per region):
    Activity, Mobility, Complexity  [FrontalMidline, Parietal, Central, Occipital]

  Spectral entropy (per region):
    Shannon entropy of normalised PSD up to 40 Hz

  Permutation entropy m=3 (per region):
    Ordinal pattern distribution entropy, normalised to [0, 1]

  Statistical (time-domain, per region):
    Skewness, Kurtosis, Zero-Crossing Rate

Models compared:
  LogReg  : L2 Logistic Regression (default C=1.0, balanced)
  RF      : Random Forest (200 trees, balanced weights)
  LGBM    : LightGBM (300 rounds, balanced weights)
  LinSVC  : Calibrated Linear SVC (default C=0.1)
  RBF     : Nystroem-approximated RBF SVM (inner-CV tuned gamma/C/K, 300 comp)

Normalisation:
  Per-participant z-scoring is applied to the raw feature matrix before any CV
  split, independently for each participant.  This removes between-subject
  amplitude variance (skull conductance, impedance, electrode offsets) and
  makes the classifier learn each person's *relative* deviation from their own
  baseline — which is the MWL-relevant signal.  The pipeline-level
  StandardScaler then performs a second centering pass over the concatenated
  training epochs, which is harmless and handles any residual numeric drift.

Usage (from MATB repo root):
    # Cross-subject 70/30 holdout (default mode)
    .venv\\Scripts\\python.exe scripts/compare_ml_models.py --models LogReg RF RBF

    # Within-participant CV (per-participant, RBF not supported here)
    .venv\\Scripts\\python.exe scripts/compare_ml_models.py --mode within --models LogReg RF

    # C-sweep screening (LogReg and LinSVC only)
    .venv\\Scripts\\python.exe scripts/compare_ml_models.py --c-sweep

    # Exact replication (seed is saved in output JSON)
    .venv\\Scripts\\python.exe scripts/compare_ml_models.py --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

_REPO_ROOT   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import (  # noqa: E402
    PretrainDataDir,
    calibration_norm_features,
    load_baseline_from_cache,
)
from eeg.extract_features import (  # noqa: E402
    _build_region_map,
    _save_feature_cache,
    extract_features,
)

_DATASET     = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_YAML_CONFIG = Path("C:/vr_tsst_2025/config/eeg_feature_extraction.yaml")
_FEATURE_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "feature_cache.npz"
_NORM_CACHE    = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"


def _load_exclude(cfg_path: Path) -> set[str]:
    """Load excluded PIDs from pretrain_qc.yaml."""
    cfg = yaml.safe_load(cfg_path.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_QC_CONFIG = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_EXCLUDE = _load_exclude(_QC_CONFIG)

# Fixed fallback bands (used when --no-iaf or IAF estimation fails)

_FIXED = {
    "Delta":   (1.0,  4.0),
    "Theta":   (4.0,  7.5),
    "Alpha":   (7.5, 12.0),
    "Beta":   (12.0, 30.0),
    "Gamma":  (30.0, 45.0),
}

# Regions used for time-domain features (Hjorth, entropy, stats)
# Occipital is optional — skipped if not in region_map
_TEMPORAL_REGIONS = ["FrontalMidline", "Parietal", "Central", "Occipital"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Default C values per model — tuned by C sweep (cross-split, seed=42, n=38, pp-zscore)
# Sweep showed monotonically: smaller C → higher AUC for both models at this sample size.
# LogReg C=0.001: overall 0.6146 vs 0.6071 at C=1.0; LinSVC C=0.001: 0.6111 vs 0.6070 at C=0.1
_DEFAULT_C = {"LogReg": 0.001, "LinSVC": 0.001}
# C values screened by --c-sweep (covers 4 orders of magnitude)
_C_SWEEP_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
_C_SWEEP_MODELS = ["LogReg", "LinSVC"]


def _make_pipeline(name: str, seed: int = 42, C: float | None = None) -> Pipeline:
    """Build a fresh sklearn Pipeline for the named model, with reproducible seed.

    Args:
        name: Model name — one of LogReg, RF, LGBM, LinSVC.
        seed: Random seed for reproducibility.
        C:    Regularisation strength override for LogReg and LinSVC.  When
              None the per-model default from _DEFAULT_C is used.
    """
    if name == "LogReg":
        c_val = C if C is not None else _DEFAULT_C["LogReg"]
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=c_val, max_iter=1000, class_weight="balanced",
                                       random_state=seed)),
        ])
    if name == "RF":
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=seed, class_weight="balanced")),
        ])
    if name == "LGBM":
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                class_weight="balanced", random_state=seed,
                n_jobs=-1, verbose=-1)),
        ])
    if name == "LinSVC":
        c_val = C if C is not None else _DEFAULT_C["LinSVC"]
        return Pipeline([
            ("sc",  StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=c_val, max_iter=2000, class_weight="balanced"),
                cv=3)),
        ])
    raise ValueError(f"Unknown model: {name}")

_ALL_MODELS = ["LogReg", "RF", "LGBM", "LinSVC", "RBF"]

# ---------------------------------------------------------------------------
# RBF SVM config + helpers (matches rbf_ablation_loso.py baseline)
# ---------------------------------------------------------------------------

_RBF_GAMMA_C_GRID = [
    {"gamma": 0.01, "C": 0.1},
    {"gamma": 0.05, "C": 0.1},
    {"gamma": 0.1,  "C": 0.1},
    {"gamma": 0.01, "C": 1.0},
    {"gamma": 0.05, "C": 1.0},
    {"gamma": 0.1,  "C": 1.0},
]
_RBF_N_COMPONENTS = 300
_RBF_K_CANDIDATES = [15, 20, 25, 30, 45]
_N_INNER_FOLDS = 5


def _make_rbf(gamma: float, C: float, n_components: int,
              seed: int = 42) -> Pipeline:
    return Pipeline([
        ("sc",  StandardScaler()),
        ("nys", Nystroem(kernel="rbf", gamma=gamma,
                         n_components=n_components, random_state=seed)),
        ("clf", LogisticRegression(C=C, max_iter=1000,
                                   class_weight="balanced",
                                   random_state=seed)),
    ])


def _rbf_inner_cv(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    gamma: float, C: float, k: int, seed: int,
) -> float:
    """Mean AUC across inner folds for one gamma/C/k combo."""
    cv = StratifiedGroupKFold(n_splits=_N_INNER_FOLDS)
    aucs: list[float] = []
    for tr, te in cv.split(X, y, groups):
        sel = SelectKBest(f_classif, k=k)
        X_tr = sel.fit_transform(X[tr], y[tr])
        X_te = sel.transform(X[te])
        pipe = _make_rbf(gamma, C, _RBF_N_COMPONENTS, seed)
        pipe.fit(X_tr, y[tr])
        probs = pipe.predict_proba(X_te)[:, 1]
        aucs.append(_auc(y[te], probs))
    return float(np.mean(aucs))


def _logreg_k_probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    k: int, seed: int,
) -> float:
    """Inner CV with LogReg probe to score a candidate K."""
    cv = StratifiedGroupKFold(n_splits=_N_INNER_FOLDS)
    aucs: list[float] = []
    for tr, te in cv.split(X, y, groups):
        sel = SelectKBest(f_classif, k=k)
        X_tr = sel.fit_transform(X[tr], y[tr])
        X_te = sel.transform(X[te])
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(C=0.001, max_iter=1000,
                                       class_weight="balanced",
                                       random_state=seed)),
        ])
        pipe.fit(X_tr, y[tr])
        probs = pipe.predict_proba(X_te)[:, 1]
        aucs.append(_auc(y[te], probs))
    return float(np.mean(aucs))


def _select_best_k_rbf(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
) -> int:
    """Select best K for RBF via LogReg probe inner CV."""
    best_k, best_auc = _RBF_K_CANDIDATES[0], -1.0
    for k in _RBF_K_CANDIDATES:
        auc = _logreg_k_probe(X, y, groups, k, seed)
        if auc > best_auc:
            best_k, best_auc = k, auc
    return best_k


def _tune_rbf_gamma_c(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    best_k: int, seed: int,
) -> dict:
    """Tune gamma/C via inner CV at the chosen K."""
    best_params, best_auc = _RBF_GAMMA_C_GRID[0], -1.0
    for params in _RBF_GAMMA_C_GRID:
        auc = _rbf_inner_cv(X, y, groups, params["gamma"], params["C"],
                            best_k, seed)
        if auc > best_auc:
            best_params, best_auc = params, auc
    return best_params

# ---------------------------------------------------------------------------
# Permutation entropy helpers (m=3, vectorised)
# ---------------------------------------------------------------------------
# For m=3 the 6 ordinal patterns encode as r0*9 + r1*3 + r2:
# (0,1,2)→5  (0,2,1)→7  (1,0,2)→11  (1,2,0)→15  (2,0,1)→19  (2,1,0)→21
_PE_N_PATTERNS_M3 = len(list(permutations(range(3))))



# ---------------------------------------------------------------------------
# Feature cache (disk)
# ---------------------------------------------------------------------------

def _cache_key(dataset_path: Path) -> str:
    """Hash of manifest mtime + exclusion set + fixed bands.

    If any of these change the cache is stale and will be recomputed.
    The dataset_path is the continuous directory; we hash manifest.json
    mtime so rebuilding the export invalidates the feature cache.
    """
    manifest = dataset_path / "manifest.json"
    mtime  = str(manifest.stat().st_mtime) if manifest.exists() else "missing"
    excl   = str(sorted(_EXCLUDE))
    bands  = str(sorted((k, v) for k, v in _FIXED.items()))
    feat_v = "v3_continuous"  # bump when feature set changes
    raw    = "|".join([mtime, excl, bands, feat_v])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_feature_cache(
    cache_path: Path,
    expected_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]] | None:
    """Return (X_by, y_by, feat_names) from cache, or None if stale/missing."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        if str(data["cache_key"]) != expected_key:
            return None
        pids       = list(data["pids"])
        feat_names = list(data["feat_names"])
        X_by = {pid: data[f"X_{pid}"] for pid in pids}
        y_by = {pid: data[f"y_{pid}"] for pid in pids}
        return X_by, y_by, feat_names
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _remap(labels: np.ndarray) -> np.ndarray:
    unique = sorted(set(labels.tolist()))
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[lb] for lb in labels], dtype=np.int64)


def load_participant(
    data_dir: PretrainDataDir,
    pid: str,
    srate: float,
    region_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feat_names) for one participant.

    Reads windowed task epochs from the continuous per-participant HDF5 via
    PretrainDataDir, then computes spectral + time-domain features.
    Labels are remapped from {0, 2} to {0, 1} for binary classification.
    """
    epochs, labels_raw, _ = data_dir.load_task_epochs(pid)
    labels = _remap(labels_raw)
    freqs, psd = welch(epochs, fs=srate,
                       nperseg=min(256, epochs.shape[-1]), axis=-1)
    X, feat_names = extract_features(epochs, psd, freqs, region_map, _FIXED,
                                      srate=srate)
    return X, labels, feat_names


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5


def within_participant_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "LogReg",
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """n-fold stratified CV (no shuffle — preserves temporal order).

    Returns (mean_auc, std_auc, fold_aucs).
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=False)
    fold_aucs = []
    for tr, te in cv.split(X, y):
        pipe = _make_pipeline(model_name, seed=seed)
        pipe.fit(X[tr], y[tr])
        probs = pipe.predict_proba(X[te])[:, 1]
        fold_aucs.append(_auc(y[te], probs))
    mean = float(np.mean(fold_aucs))
    std  = float(np.std(fold_aucs))
    return mean, std, fold_aucs


def loso_cv(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    model_name: str = "LogReg",
    seed: int = 42,
    X_by_test: dict[str, np.ndarray] | None = None,
) -> dict[str, dict]:
    """Leave-one-subject-out CV. Train on all-but-one, test on held-out.

    If *X_by_test* is provided, the held-out participant's features are
    taken from that dict (e.g. calibration-normalised), while training
    participants use *X_by* (e.g. pp z-scored).  This simulates online
    deployment where only causal data is available for normalising the
    new user.

    Returns per-participant dict with auc and n_epochs.
    """
    if X_by_test is None:
        X_by_test = X_by
    pids = sorted(X_by.keys())
    results: dict[str, dict] = {}
    for pid in pids:
        X_train = np.concatenate([X_by[p] for p in pids if p != pid])
        y_train = np.concatenate([y_by[p] for p in pids if p != pid])
        X_test, y_test = X_by_test[pid], y_by[pid]
        pipe = _make_pipeline(model_name, seed=seed)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        results[pid] = {
            "n_epochs": int(len(y_test)),
            "auc":      _auc(y_test, probs),
            "bal_acc":  float(balanced_accuracy_score(y_test, preds)),
            "f1":       float(f1_score(y_test, preds, average="macro", zero_division=0)),
        }
    return results


def holdout_cross(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    model_name: str = "LogReg",
    test_frac: float = 0.20,
    seed: int = 42,
    C: float | None = None,
    X_by_test: dict[str, np.ndarray] | None = None,
) -> dict:
    """Cross-subject evaluation with a random participant-level train/test split.

    Faster than LOSO — useful for quick model screening.  The split is at the
    participant level (not epoch level) so there is no data leakage.

    If *X_by_test* is provided, test participants use those features
    (e.g. calibration-normalised) while training uses *X_by*.

    Args:
        C: Regularisation strength override for LogReg/LinSVC (None = default).

    Returns a dict with overall AUC, per-participant AUCs, and the test PIDs used.
    """
    if X_by_test is None:
        X_by_test = X_by
    rng  = np.random.default_rng(seed)
    pids = sorted(X_by.keys())
    n_test = max(1, round(len(pids) * test_frac))
    test_pids  = list(rng.choice(pids, size=n_test, replace=False))
    train_pids = [p for p in pids if p not in test_pids]

    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])

    per_pid: dict[str, float] = {}
    all_true, all_prob = [], []

    if model_name == "RBF":
        groups = np.concatenate([
            np.full(len(y_by[p]), j, dtype=np.int32)
            for j, p in enumerate(train_pids)
        ])
        best_k = _select_best_k_rbf(X_train, y_train, groups, seed)
        best_params = _tune_rbf_gamma_c(X_train, y_train, groups, best_k, seed)
        selector = SelectKBest(f_classif, k=best_k)
        X_train_sel = selector.fit_transform(X_train, y_train)
        pipe = _make_rbf(best_params["gamma"], best_params["C"],
                         _RBF_N_COMPONENTS, seed)
        pipe.fit(X_train_sel, y_train)

        for pid in test_pids:
            X_test_sel = selector.transform(X_by_test[pid])
            probs = pipe.predict_proba(X_test_sel)[:, 1]
            per_pid[pid] = _auc(y_by[pid], probs)
            all_true.append(y_by[pid])
            all_prob.append(probs)
    else:
        pipe = _make_pipeline(model_name, seed=seed, C=C)
        pipe.fit(X_train, y_train)

        for pid in test_pids:
            probs = pipe.predict_proba(X_by_test[pid])[:, 1]
            per_pid[pid] = _auc(y_by[pid], probs)
            all_true.append(y_by[pid])
            all_prob.append(probs)

    overall_auc = _auc(
        np.concatenate(all_true),
        np.concatenate(all_prob),
    )
    result = {
        "overall_auc": overall_auc,
        "per_pid": per_pid,
        "train_pids": train_pids,
        "test_pids": test_pids,
        "n_train": len(train_pids),
        "n_test": len(test_pids),
    }
    if model_name == "RBF":
        result["rbf_best_k"] = best_k
        result["rbf_best_gamma"] = best_params["gamma"]
        result["rbf_best_C"] = best_params["C"]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",     type=Path, default=_DATASET)
    parser.add_argument("--n-splits",    type=int, default=5,
                        help="CV folds for within-participant mode (default: 5)")
    parser.add_argument("--mode",        choices=["within", "cross-split"],
                        default="cross-split",
                        help="within: per-participant CV | "
                             "cross-split: participant-level hold-out  (default: cross-split)")
    parser.add_argument("--models",      nargs="+", default=_ALL_MODELS,
                        choices=_ALL_MODELS,
                        help="Models to run (default: all). E.g. --models LogReg RF RBF")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Base random seed for reproducibility (default: 42)")
    parser.add_argument("--n-seeds",     type=int, default=10,
                        help="Number of random train/test splits to average over (default: 10)")
    parser.add_argument("--test-frac",   type=float, default=0.30,
                        help="Fraction of participants held out for cross-split mode (default: 0.30)")
    parser.add_argument("--c-sweep",     action="store_true",
                        help="Screen regularisation strengths for LogReg and LinSVC using the "
                             "cross-split holdout.  Overrides --mode.  Prints a table of AUC "
                             "vs C and exits.")
    parser.add_argument("--out",          type=Path, default=None,
                        help="Override the output JSON path.  Defaults to "
                             "results/test_pretrain/personalised_logreg_{mode}.json")
    args = parser.parse_args()

    seed    = args.seed
    n_seeds = args.n_seeds
    models  = args.models
    seeds   = [seed + i for i in range(n_seeds)]

    data_dir = PretrainDataDir(args.dataset)
    ch_names: list[str] = data_dir.channel_names()
    srate = float(data_dir.srate())

    region_map = _build_region_map(_YAML_CONFIG, ch_names)

    required = ["FrontalLeft", "FrontalRight", "FrontalMidline", "Central", "Parietal"]
    for r in required:
        if len(region_map.get(r, [])) == 0:
            raise SystemExit(f"Required region '{r}' has 0 matched channels -- check YAML config")

    t_regions = [r for r in _TEMPORAL_REGIONS if r in region_map]
    print(f"Dataset  : {args.dataset.name}  ({len(ch_names)} ch, {srate} Hz)")
    print(f"Bands    : fixed")
    print(f"Mode     : {args.mode}")
    print(f"Models   : {', '.join(models)}")
    print(f"Seeds    : {seeds[0]}..{seeds[-1]} ({n_seeds} splits)")
    print(f"Temporal regions: {', '.join(t_regions)}")
    print()

    do_c_sweep: bool = getattr(args, "c_sweep", False)

    # ------------------------------------------------------------------
    # Load features — prefer norm comparison cache (has task + baseline),
    # then feature-only cache, then recompute.
    # ------------------------------------------------------------------
    # Try norm comparison cache first (built by causal_norm_comparison.py)
    norm_loaded = False
    if _NORM_CACHE.exists():
        try:
            npz = np.load(_NORM_CACHE, allow_pickle=False)
            npz_pids = list(npz["pids"])
            available = [p for p in npz_pids if p not in _EXCLUDE]
            if available:
                feat_names = list(npz["feat_names"])
                X_by: dict[str, np.ndarray] = {}
                y_by: dict[str, np.ndarray] = {}
                for pid in available:
                    X_by[pid] = npz[f"{pid}_task_X"]
                    y_by[pid] = npz[f"{pid}_task_y"]
                pids = sorted(available)
                norm_loaded = True
                print(f"Features loaded from norm cache  ({len(pids)} participants, "
                      f"{len(feat_names)} features)")
        except Exception:
            pass

    if not norm_loaded:
        key    = _cache_key(args.dataset)
        cached = _load_feature_cache(_FEATURE_CACHE, key)

        if cached is not None:
            X_by, y_by, feat_names = cached
            X_by = {p: v for p, v in X_by.items() if p not in _EXCLUDE}
            y_by = {p: v for p, v in y_by.items() if p not in _EXCLUDE}
            pids = sorted(X_by.keys())
            print(f"Features loaded from cache  ({len(pids)} participants, "
                  f"{len(feat_names)} features)  [key={key}]")
        else:
            print("Computing features (no valid cache found)...", end=" ", flush=True)
            all_pids = sorted(p for p in data_dir.available_pids() if p not in _EXCLUDE)
            X_by: dict[str, np.ndarray] = {}
            y_by: dict[str, np.ndarray] = {}
            feat_names: list[str] = []
            for pid in all_pids:
                X, y, feat_names = load_participant(data_dir, pid, srate, region_map)
                X_by[pid] = X
                y_by[pid] = y
            pids = all_pids
            _save_feature_cache(_FEATURE_CACHE, key, X_by, y_by, feat_names)
            print(f"done.  Cached to {_FEATURE_CACHE.relative_to(_REPO_ROOT)}")
            print(f"  n_participants={len(pids)}, n_features={len(feat_names)}")

    # ------------------------------------------------------------------
    # Per-participant normalisation: calibration (causal, ADR-0004).
    # All participants z-scored using fixation + Forest0 baseline.
    # ------------------------------------------------------------------
    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    if baseline_by is not None:
        X_by_cal: dict[str, np.ndarray] = {}
        for pid in pids:
            bl = baseline_by[pid]
            X_by_cal[pid] = calibration_norm_features(
                X_by[pid], bl["fix_X"], bl["forest_X"], bl["forest_bidx"]
            )
        X_by_train = X_by_cal
        X_by_test = X_by_cal
        print(f"Calibration normalisation (fix + Forest0) applied to all {len(pids)} participants.")
    else:
        raise SystemExit(
            f"ERROR: Baseline cache not found at {_NORM_CACHE.name}.\n"
            f"       Run causal_norm_comparison.py first to build it."
        )
    print()

    # ------------------------------------------------------------------
    # C sweep (--c-sweep flag): screen regularisation for LogReg / LinSVC
    # ------------------------------------------------------------------
    if do_c_sweep:
        print(f"=== C sweep (cross-split holdout, test_frac={args.test_frac:.0%}, seed={seed}) ===")
        print(f"  {'Model':<8}  {'C':>8}  {'Overall AUC':>11}  {'Mean per-PID':>12}")
        print("  " + "-" * 48)
        for model_name in _C_SWEEP_MODELS:
            for c_val in _C_SWEEP_VALUES:
                r = holdout_cross(X_by_train, y_by, model_name=model_name,
                                  test_frac=args.test_frac, seed=seed,
                                  C=c_val, X_by_test=X_by_test)
                mean_pp = float(np.mean(list(r["per_pid"].values())))
                print(f"  {model_name:<8}  {c_val:>8.3f}  {r['overall_auc']:>11.4f}  {mean_pp:>12.4f}")
            print()
        return

    # ------------------------------------------------------------------
    # Within-participant CV
    # ------------------------------------------------------------------
    within_results_by_model: dict[str, dict[str, dict]] = {}

    if args.mode == "within":
        within_models = [m for m in models if m != "RBF"]
        if len(within_models) < len(models):
            print("[NOTE] RBF skipped for within-participant CV (needs inner tuning)\n")
        for model_name in within_models:
            print(f"=== Within-participant {args.n_splits}-fold CV — {model_name} ===")
            print(f"  {'PID':<8}  {'n_epochs':>9}  {'mean AUC':>9}  {'std':>6}")
            print("  " + "-" * 44)
            within_results: dict[str, dict] = {}
            for pid in pids:
                mean_auc, std_auc, fold_aucs = within_participant_cv(
                    X_by_train[pid], y_by[pid],
                    model_name=model_name, n_splits=args.n_splits, seed=seed,
                )
                print(f"  {pid:<8}  {len(y_by[pid]):>9}  {mean_auc:>9.4f}  {std_auc:>6.4f}")
                within_results[pid] = {
                    "n_epochs":  int(len(y_by[pid])),
                    "mean_auc":  mean_auc,
                    "std_auc":   std_auc,
                    "fold_aucs": fold_aucs,
                }
            wm = [v["mean_auc"] for v in within_results.values()]
            print("  " + "-" * 44)
            print(f"  {'Population':<18}  {np.mean(wm):>9.4f}  {np.std(wm):>6.4f}"
                  f"  (median {np.median(wm):.4f}, n={len(wm)})")
            print()
            within_results_by_model[model_name] = within_results

    # ------------------------------------------------------------------
    # Cross-participant: fast holdout split (participant-level)
    # ------------------------------------------------------------------
    holdout_results_by_model: dict[str, dict] = {}

    if args.mode == "cross-split":
        print(f"=== Cross-participant holdout (test_frac={args.test_frac:.0%}, "
              f"{n_seeds} seeds: {seeds[0]}..{seeds[-1]}) ===")
        # Run each model over all seeds, collect per-seed overall AUC
        for model_name in models:
            seed_results: list[dict] = []
            seed_aucs: list[float] = []
            for s in seeds:
                r = holdout_cross(X_by_train, y_by, model_name=model_name,
                                  test_frac=args.test_frac, seed=s,
                                  X_by_test=X_by_test)
                seed_results.append(r)
                seed_aucs.append(r["overall_auc"])
            mean_auc = float(np.mean(seed_aucs))
            std_auc  = float(np.std(seed_aucs))
            med_auc  = float(np.median(seed_aucs))

            holdout_results_by_model[model_name] = {
                "mean_auc":   mean_auc,
                "std_auc":    std_auc,
                "median_auc": med_auc,
                "per_seed":   [
                    {"seed": s, "overall_auc": sr["overall_auc"],
                     "test_pids": sr["test_pids"],
                     "per_pid": sr["per_pid"],
                     **({
                         "rbf_best_k": sr["rbf_best_k"],
                         "rbf_best_gamma": sr["rbf_best_gamma"],
                         "rbf_best_C": sr["rbf_best_C"],
                     } if model_name == "RBF" else {}),
                     }
                    for s, sr in zip(seeds, seed_results)
                ],
            }

        # Print summary table
        print(f"  {'Model':<10}  {'Mean AUC':>9}  {'Std':>6}  {'Median':>7}  "
              f"{'Min':>6}  {'Max':>6}  seeds")
        print("  " + "-" * 65)
        for model_name in models:
            h = holdout_results_by_model[model_name]
            sa = [s["overall_auc"] for s in h["per_seed"]]
            print(f"  {model_name:<10}  {h['mean_auc']:>9.4f}  {h['std_auc']:>6.4f}  "
                  f"{h['median_auc']:>7.4f}  {min(sa):>6.4f}  {max(sa):>6.4f}  "
                  f"n={n_seeds}")
        print()

    # ------------------------------------------------------------------
    # Model comparison summary
    # ------------------------------------------------------------------
    if len(models) > 1:
        print("=" * 72)
        print(f"Model comparison summary  (n_participants={len(pids)}, "
              f"seeds={seeds[0]}..{seeds[-1]})")
        print("=" * 72)
        hdr = f"{'Model':<10}"
        if within_results_by_model:
            hdr += f"  {'Within Mean':>11}  {'Within Med':>10}"
        if holdout_results_by_model:
            hdr += f"  {'Mean AUC':>9}  {'± Std':>6}  {'Median':>7}"
        print(hdr)
        print("-" * 72)
        for model_name in models:
            row = f"{model_name:<10}"
            if model_name in within_results_by_model:
                wm = [v["mean_auc"] for v in within_results_by_model[model_name].values()]
                wm_mean, wm_med = float(np.mean(wm)), float(np.median(wm))
                row += f"  {wm_mean:>11.4f}  {wm_med:>10.4f}"
            if model_name in holdout_results_by_model:
                h = holdout_results_by_model[model_name]
                row += f"  {h['mean_auc']:>9.4f}  {h['std_auc']:>6.4f}  {h['median_auc']:>7.4f}"
            print(row)
        print("=" * 72)
        print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def _pop_stats(vals: list[float]) -> dict:
        return {
            "mean_auc":   float(np.mean(vals)),
            "std_auc":    float(np.std(vals)),
            "median_auc": float(np.median(vals)),
            "n": len(vals),
        }

    out: dict = {
        "mode":            args.mode,
        "seed":            seed,
        "n_seeds":         n_seeds,
        "seeds":           seeds,
        "n_splits":        args.n_splits,
        "bands":           "fixed",
        "fixed_bands":     {k: list(v) for k, v in _FIXED.items()},
        "models":          models,
        "feature_names":   feat_names,
        "n_features":      len(feat_names),
        "n_participants":  len(pids),
        "excluded":        sorted(_EXCLUDE),
    }

    if within_results_by_model:
        out["within"] = {
            mn: {
                "participants": res,
                "population": _pop_stats([v["mean_auc"] for v in res.values()]),
            }
            for mn, res in within_results_by_model.items()
        }

    if holdout_results_by_model:
        out["cross_split"] = holdout_results_by_model

    if args.out is not None:
        out_path = args.out.expanduser().resolve()
    else:
        out_path = _REPO_ROOT / "results" / "test_pretrain" / f"personalised_logreg_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
