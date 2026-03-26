"""RBF ablation LOSO: squeeze every drop from Nystroem-RBF cross-participant MWL.

Runs 10 experiments sequentially, each changing exactly ONE thing from a
shared baseline (except H/I combos). Results are saved incrementally to a
single JSON file so nothing is lost if the script is killed overnight.

Experiments
-----------
  A   baseline     — RBF-only, same grid as ensemble_loso.py (6 configs, LogReg K probe, 300 comp)
  B   wide_gamma   — 10 gammas × 6 Cs = 60 configs (finer gamma sweep + wider C)
  B2  wide_C       — base 3 gammas × 6 Cs = 18 configs (isolate wider C effect)
  C   rbf_k_probe  — use RBF itself (not LogReg) to select best K
  D   nys_500      — Nystroem n_components=500
  E   nys_800      — Nystroem n_components=800
  F   fine_k       — k ∈ {10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45}
  G   mi_select    — mutual_info_classif instead of f_classif for SelectKBest
  H   combined     — wide_gamma + rbf_k_probe + nys_500 + fine_k (f_classif)
  I   combined_mi  — H + mutual_info_classif

Each experiment is a full 28-fold LOSO with inner StratifiedGroupKFold(5).
NOTE: Uses the original hardcoded _EXCLUDE set (19 participants) — this is
a historical analysis on the original 28-participant dataset.  The current
pipeline uses 40 participants (see config/pretrain_qc.yaml).
Incremental save after every experiment completes.

Usage:
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/rbf_ablation_loso.py
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/rbf_ablation_loso.py --only B D H
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/rbf_ablation_loso.py --out results/test_pretrain/rbf_ablation_custom.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants  (must match personalised_logreg.py for cache compatibility)
# ---------------------------------------------------------------------------

_REPO_ROOT     = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import load_baseline_from_cache, prepare_mixed_norm  # noqa: E402

_DATASET       = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_FEATURE_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "feature_cache.npz"
_NORM_CACHE    = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"

_EXCLUDE = {
    "P16", "P21", "P27", "P34", "P37", "P43", "P45", "P31", "P13",
    "P04", "P06", "P09", "P20", "P23", "P33", "P35", "P39", "P44", "P46",
}

_FIXED = {
    "Delta": (1.0,  4.0),
    "Theta": (4.0,  7.5),
    "Alpha": (7.5, 12.0),
    "Beta":  (12.0, 30.0),
    "Gamma": (30.0, 45.0),
}

_TEMPORAL_REGIONS = ["FrontalMidline", "Parietal", "Central", "Occipital"]

_N_INNER_FOLDS = 5
_SEED          = 42


# ===================================================================
# Experiment definitions — each is a dict describing what changes
# ===================================================================

def _baseline_config() -> dict:
    """Shared baseline = RBF-only LOSO from ensemble_loso.py."""
    return {
        "gamma_C_grid": [
            {"gamma": 0.01, "C": 0.1},
            {"gamma": 0.05, "C": 0.1},
            {"gamma": 0.1,  "C": 0.1},
            {"gamma": 0.01, "C": 1.0},
            {"gamma": 0.05, "C": 1.0},
            {"gamma": 0.1,  "C": 1.0},
        ],
        "n_components": 300,
        "k_candidates": [15, 20, 25, 30, 45],
        "k_probe": "LogReg",          # model used for K selection
        "selector": "f_classif",      # feature selection scorer
    }


_WIDE_GAMMA_GRID = [
    {"gamma": g, "C": c}
    for g in [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]
    for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
]

_WIDE_C_GRID = [
    {"gamma": g, "C": c}
    for g in [0.01, 0.05, 0.1]
    for c in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
]

_FINE_K = [10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45]


EXPERIMENTS: dict[str, dict] = {}

# A — baseline (no changes)
EXPERIMENTS["A"] = {
    "label": "baseline",
    "description": "RBF-only, same grid as ensemble_loso (6 configs, LogReg K probe, 300 comp, f_classif)",
    "changes": {},
}

# B — wider gamma grid (10 gammas × 6 Cs = 60 configs)
EXPERIMENTS["B"] = {
    "label": "wide_gamma",
    "description": "10 gammas × 6 Cs = 60 configs (finer gamma sweep + wider C)",
    "changes": {"gamma_C_grid": _WIDE_GAMMA_GRID},
}

# B2 — wider C only (isolate the C dimension)
EXPERIMENTS["B2"] = {
    "label": "wide_C",
    "description": "Base 3 gammas × 6 Cs = 18 configs (isolate wider C effect)",
    "changes": {"gamma_C_grid": _WIDE_C_GRID},
}

# C — use RBF itself as K probe
EXPERIMENTS["C"] = {
    "label": "rbf_k_probe",
    "description": "Use RBF (not LogReg) to select best K — gamma=0.05, C=1.0 probe",
    "changes": {"k_probe": "RBF"},
}

# D — Nystroem 500 components
EXPERIMENTS["D"] = {
    "label": "nys_500",
    "description": "Nystroem n_components=500 (baseline=300)",
    "changes": {"n_components": 500},
}

# E — Nystroem 800 components
EXPERIMENTS["E"] = {
    "label": "nys_800",
    "description": "Nystroem n_components=800 (baseline=300)",
    "changes": {"n_components": 800},
}

# F — finer K grid
EXPERIMENTS["F"] = {
    "label": "fine_k",
    "description": "k ∈ {10,15,18,20,22,25,28,30,35,40,45} (baseline has 5 values)",
    "changes": {"k_candidates": _FINE_K},
}

# G — mutual information for feature selection
EXPERIMENTS["G"] = {
    "label": "mi_select",
    "description": "mutual_info_classif instead of f_classif for SelectKBest",
    "changes": {"selector": "mutual_info_classif"},
}

# H — combine all changes (conservative: nys500)
EXPERIMENTS["H"] = {
    "label": "combined",
    "description": "wide_gamma + rbf_k_probe + nys_500 + fine_k (f_classif — MI only if G wins)",
    "changes": {
        "gamma_C_grid": _WIDE_GAMMA_GRID,
        "k_probe": "RBF",
        "n_components": 500,
        "k_candidates": _FINE_K,
    },
}

# I — H + mutual info (if G showed MI helps)
EXPERIMENTS["I"] = {
    "label": "combined_mi",
    "description": "H + mutual_info_classif (full everything combo)",
    "changes": {
        "gamma_C_grid": _WIDE_GAMMA_GRID,
        "k_probe": "RBF",
        "n_components": 500,
        "k_candidates": _FINE_K,
        "selector": "mutual_info_classif",
    },
}


# ---------------------------------------------------------------------------
# Cache helpers  (identical to ensemble_loso.py)
# ---------------------------------------------------------------------------

def _cache_key(dataset_path: Path) -> str:
    manifest = dataset_path / "manifest.json"
    mtime = str(manifest.stat().st_mtime) if manifest.exists() else "missing"
    excl  = str(sorted(_EXCLUDE))
    bands = str(sorted((k, v) for k, v in _FIXED.items()))
    feat_v = "v3_continuous"
    raw   = "|".join([mtime, excl, bands, feat_v])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_feature_cache(
    cache_path: Path,
    expected_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]] | None:
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
# Small helpers
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5


def _pop_stats(vals: list[float]) -> dict:
    return {
        "mean":   round(float(np.mean(vals)),   4),
        "std":    round(float(np.std(vals)),    4),
        "median": round(float(np.median(vals)), 4),
        "n":      len(vals),
    }


def _jsonable(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# RBF pipeline factory
# ---------------------------------------------------------------------------

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


def _make_logreg_probe(seed: int = 42) -> Pipeline:
    """LogReg(C=0.001) used for K selection in the baseline."""
    return Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(C=0.001, max_iter=1000,
                                   class_weight="balanced",
                                   random_state=seed)),
    ])


# ---------------------------------------------------------------------------
# Feature-selection scorer wrapper
# ---------------------------------------------------------------------------

def _get_selector(scorer_name: str, k: int, seed: int) -> SelectKBest:
    """Return a SelectKBest configured for the requested scorer."""
    if scorer_name == "mutual_info_classif":
        # MI needs a wrapper to pass random_state
        def _mi(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return mutual_info_classif(X, y, random_state=seed, n_neighbors=5)
        return SelectKBest(_mi, k=k)
    return SelectKBest(f_classif, k=k)


# ---------------------------------------------------------------------------
# Inner-CV routines  (RBF-only, parameterised by experiment config)
# ---------------------------------------------------------------------------

def _inner_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    gamma: float,
    C: float,
    k: int,
    cfg: dict,
    seed: int,
) -> float:
    """Mean AUC across inner folds.  SelectKBest fit inside each fold."""
    cv = StratifiedGroupKFold(n_splits=_N_INNER_FOLDS)
    aucs: list[float] = []
    n_comp = cfg["n_components"]
    scorer = cfg["selector"]
    for tr, te in cv.split(X, y, groups):
        sel  = _get_selector(scorer, k, seed)
        X_tr = sel.fit_transform(X[tr], y[tr])
        X_te = sel.transform(X[te])
        pipe = _make_rbf(gamma, C, n_comp, seed)
        pipe.fit(X_tr, y[tr])
        probs = pipe.predict_proba(X_te)[:, 1]
        aucs.append(_auc(y[te], probs))
    return float(np.mean(aucs))


def _inner_cv_score_logreg(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
    cfg: dict,
    seed: int,
) -> float:
    """Inner CV with LogReg probe — used when k_probe == 'LogReg'."""
    cv = StratifiedGroupKFold(n_splits=_N_INNER_FOLDS)
    aucs: list[float] = []
    scorer = cfg["selector"]
    for tr, te in cv.split(X, y, groups):
        sel  = _get_selector(scorer, k, seed)
        X_tr = sel.fit_transform(X[tr], y[tr])
        X_te = sel.transform(X[te])
        pipe = _make_logreg_probe(seed)
        pipe.fit(X_tr, y[tr])
        probs = pipe.predict_proba(X_te)[:, 1]
        aucs.append(_auc(y[te], probs))
    return float(np.mean(aucs))


def _select_best_k(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict,
    seed: int,
) -> int:
    """Select K using the probe model specified in cfg."""
    k_candidates = cfg["k_candidates"]
    k_probe = cfg["k_probe"]
    best_k, best_auc = k_candidates[0], -1.0

    for k in k_candidates:
        if k_probe == "LogReg":
            auc = _inner_cv_score_logreg(X, y, groups, k, cfg, seed)
        else:
            # Use RBF with a fixed mid-range config as probe
            auc = _inner_cv_score(X, y, groups,
                                  gamma=0.05, C=1.0, k=k,
                                  cfg=cfg, seed=seed)
        if auc > best_auc:
            best_k, best_auc = k, auc
    return best_k


def _tune_rbf(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    best_k: int,
    cfg: dict,
    seed: int,
) -> dict:
    """Tune gamma and C via inner CV at the chosen K."""
    grid = cfg["gamma_C_grid"]
    best_params: dict | None = None
    best_auc = -1.0
    for params in grid:
        auc = _inner_cv_score(
            X, y, groups, params["gamma"], params["C"], best_k, cfg, seed)
        if auc > best_auc:
            best_params, best_auc = params, auc
    assert best_params is not None
    return {"params": best_params, "inner_auc": round(best_auc, 4)}


# ---------------------------------------------------------------------------
# Single LOSO fold
# ---------------------------------------------------------------------------

def _run_fold(
    pid: str,
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    feat_names: list[str],
    cfg: dict,
    seed: int,
    X_by_test: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run one LOSO fold for one held-out participant."""
    if X_by_test is None:
        X_by_test = X_by
    pids = sorted(X_by.keys())
    train_pids = [p for p in pids if p != pid]
    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])
    groups  = np.concatenate([
        np.full(len(y_by[p]), j, dtype=np.int32)
        for j, p in enumerate(train_pids)
    ])
    X_test, y_test = X_by_test[pid], y_by[pid]

    # 1. Select best K
    best_k = _select_best_k(X_train, y_train, groups, cfg, seed)

    # 2. Tune gamma/C at best K
    tuned = _tune_rbf(X_train, y_train, groups, best_k, cfg, seed)

    # 3. Refit on full train, predict test
    selector = _get_selector(cfg["selector"], best_k, seed)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    selected_mask  = selector.get_support()
    selected_feats = [f for f, m in zip(feat_names, selected_mask) if m]

    pipe = _make_rbf(tuned["params"]["gamma"], tuned["params"]["C"],
                     cfg["n_components"], seed)
    pipe.fit(X_train_sel, y_train)
    probs = pipe.predict_proba(X_test_sel)[:, 1]
    preds = pipe.predict(X_test_sel)

    return {
        "n_epochs":     int(len(y_test)),
        "auc":          _auc(y_test, probs),
        "bal_acc":      float(balanced_accuracy_score(y_test, preds)),
        "f1":           float(f1_score(y_test, preds,
                                       average="macro", zero_division=0)),
        "best_k":       best_k,
        "best_gamma":   tuned["params"]["gamma"],
        "best_C":       tuned["params"]["C"],
        "inner_auc":    tuned["inner_auc"],
        "selected_features": selected_feats,
    }


# ---------------------------------------------------------------------------
# Run one full experiment (28 LOSO folds, original dataset)
# ---------------------------------------------------------------------------

def run_experiment(
    exp_key: str,
    exp_def: dict,
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    feat_names: list[str],
    seed: int,
    X_by_test: dict[str, np.ndarray] | None = None,
) -> dict:
    """Run a single experiment (all LOSO folds) and return results dict."""
    base_cfg = _baseline_config()
    for k, v in exp_def["changes"].items():
        base_cfg[k] = v

    pids = sorted(X_by.keys())
    n = len(pids)

    print(f"\n{'='*70}")
    print(f"  Experiment {exp_key}: {exp_def['label']}")
    print(f"  {exp_def['description']}")
    print(f"  grid_size={len(base_cfg['gamma_C_grid'])}  "
          f"n_comp={base_cfg['n_components']}  "
          f"k_probe={base_cfg['k_probe']}  "
          f"selector={base_cfg['selector']}  "
          f"k_cands={base_cfg['k_candidates']}")
    print(f"{'='*70}")

    fold_results: dict[str, dict] = {}
    t0 = time.perf_counter()

    for i, pid in enumerate(pids):
        tf = time.perf_counter()
        res = _run_fold(pid, X_by, y_by, feat_names, base_cfg, seed,
                        X_by_test=X_by_test)
        dt = time.perf_counter() - tf
        res["time_s"] = round(dt, 1)
        fold_results[pid] = res

        print(f"  [{i+1:2d}/{n}] {pid:<6}  "
              f"AUC {res['auc']:.4f}  "
              f"BalAcc {res['bal_acc']:.3f}  "
              f"g={res['best_gamma']:.3f} C={res['best_C']:.1f}  "
              f"k={res['best_k']}  [{dt:.0f}s]",
              flush=True)

    total_s = time.perf_counter() - t0

    # --- summary ---
    aucs = [v["auc"]     for v in fold_results.values()]
    bals = [v["bal_acc"] for v in fold_results.values()]
    f1s  = [v["f1"]      for v in fold_results.values()]
    ks   = [v["best_k"]  for v in fold_results.values()]
    gs   = [v["best_gamma"] for v in fold_results.values()]
    cs   = [v["best_C"]  for v in fold_results.values()]

    k_counts = Counter(ks)
    g_counts = Counter(gs)
    c_counts = Counter(cs)

    summary = {
        "experiment":    exp_key,
        "label":         exp_def["label"],
        "description":   exp_def["description"],
        "config": {
            "gamma_C_grid":  [
                {k: _jsonable(v) for k, v in p.items()}
                for p in base_cfg["gamma_C_grid"]
            ],
            "n_components":  base_cfg["n_components"],
            "k_candidates":  base_cfg["k_candidates"],
            "k_probe":       base_cfg["k_probe"],
            "selector":      base_cfg["selector"],
        },
        "population_auc":    _pop_stats(aucs),
        "population_balacc": _pop_stats(bals),
        "population_f1":     _pop_stats(f1s),
        "k_distribution":    {str(k): c for k, c in sorted(k_counts.items())},
        "gamma_distribution": {str(g): c for g, c in sorted(g_counts.items())},
        "C_distribution":    {str(c): c_ for c, c_ in sorted(c_counts.items())},
        "participants":      fold_results,
        "total_time_min":    round(total_s / 60, 1),
    }

    print(f"\n  --- Experiment {exp_key} summary ---")
    print(f"  AUC      mean={np.mean(aucs):.4f}  "
          f"std={np.std(aucs):.4f}  median={np.median(aucs):.4f}")
    print(f"  BalAcc   mean={np.mean(bals):.4f}  median={np.median(bals):.4f}")
    print(f"  F1       mean={np.mean(f1s):.4f}  median={np.median(f1s):.4f}")
    k_str = ", ".join(f"k={k}: {c}" for k, c in sorted(k_counts.items()))
    print(f"  K dist:  {k_str}")
    g_str = ", ".join(f"g={g}: {c}" for g, c in sorted(g_counts.items()))
    print(f"  γ dist:  {g_str}")
    print(f"  Time:    {total_s / 60:.1f} min")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=_DATASET)
    ap.add_argument("--seed",    type=int,  default=_SEED)
    ap.add_argument("--only",    nargs="*", default=None,
                    help="Run only listed experiments, e.g. --only B D H")
    ap.add_argument("--no-pp-zscore", action="store_true",
                    help="Skip per-participant z-scoring (online-compatible mode)")
    ap.add_argument("--force-pp-zscore", action="store_true",
                    help="Force full-session pp z-score even if baseline cache exists "
                         "(reproduces the optimistic upper-bound condition)")
    ap.add_argument("--out",     type=Path,
                    default=_REPO_ROOT / "results" / "test_pretrain" / "rbf_ablation.json")
    args = ap.parse_args()

    seed = args.seed
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which experiments to run
    if args.only:
        exp_keys = [k.upper() for k in args.only]
        for k in exp_keys:
            if k not in EXPERIMENTS:
                raise SystemExit(f"Unknown experiment key: {k}. "
                                 f"Valid: {sorted(EXPERIMENTS.keys())}")
    else:
        exp_keys = sorted(EXPERIMENTS.keys())

    print(f"RBF Ablation LOSO  (seed={seed})")
    print(f"Experiments: {', '.join(exp_keys)}")
    print(f"Output:      {out_path}")
    print()

    # ---- load features from existing cache ----
    key    = _cache_key(args.dataset)
    cached = _load_feature_cache(_FEATURE_CACHE, key)
    if cached is None:
        raise SystemExit(
            f"Feature cache not found or stale ({_FEATURE_CACHE}).\n"
            "Run personalised_logreg.py first to build the cache.")
    X_by_raw, y_by_raw, feat_names = cached
    X_by_raw = {p: v for p, v in X_by_raw.items() if p not in _EXCLUDE}
    y_by_raw = {p: v for p, v in y_by_raw.items() if p not in _EXCLUDE}
    pids = sorted(X_by_raw.keys())
    print(f"Loaded {len(pids)} participants, {len(feat_names)} features  "
          f"[cache key={key}]")

    # ---- per-participant normalisation (mixed: causal for test) ----
    X_by: dict[str, np.ndarray] = {}
    y_by: dict[str, np.ndarray] = {}
    X_by_test: dict[str, np.ndarray] | None = None

    baseline_by = load_baseline_from_cache(_NORM_CACHE, pids)
    if args.force_pp_zscore:
        for pid in pids:
            sc = StandardScaler()
            X_by[pid] = sc.fit_transform(X_by_raw[pid])
            y_by[pid] = y_by_raw[pid]
        print("Per-participant z-scoring FORCED (--force-pp-zscore).")
    elif baseline_by is not None and not args.no_pp_zscore:
        X_by_raw = {pid: X_by_raw[pid] for pid in pids}
        X_by, X_by_test = prepare_mixed_norm(X_by_raw, baseline_by)
        for pid in pids:
            y_by[pid] = y_by_raw[pid]
        print("Mixed normalisation: pp z-score (train) + calibration (test).")
    elif args.no_pp_zscore:
        for pid in pids:
            X_by[pid] = X_by_raw[pid].copy()
            y_by[pid] = y_by_raw[pid]
        print("Per-participant z-scoring SKIPPED (--no-pp-zscore).")
    else:
        for pid in pids:
            sc = StandardScaler()
            X_by[pid] = sc.fit_transform(X_by_raw[pid])
            y_by[pid] = y_by_raw[pid]
        print("Per-participant z-scoring applied (baseline cache not found).")

    # ---- load any previous results (resume support) ----
    all_results: dict[str, dict] = {}
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if isinstance(prev, dict) and "experiments" in prev:
                all_results = prev["experiments"]
                print(f"Loaded {len(all_results)} previous experiment(s) "
                      f"from {out_path.name}")
        except Exception:
            pass

    # ---- run experiments sequentially ----
    t_grand = time.perf_counter()

    for exp_key in exp_keys:
        if exp_key in all_results:
            print(f"\n  Skipping experiment {exp_key} "
                  f"({EXPERIMENTS[exp_key]['label']}) — already in output file.")
            continue

        summary = run_experiment(
            exp_key, EXPERIMENTS[exp_key],
            X_by, y_by, list(feat_names), seed,
            X_by_test=X_by_test)
        all_results[exp_key] = summary

        # ---- incremental save after each experiment ----
        _save(out_path, all_results, pids, feat_names, seed,
              time.perf_counter() - t_grand)

    t_grand = time.perf_counter() - t_grand

    # ---- final comparison table ----
    print(f"\n{'='*70}")
    print(f"  ABLATION COMPARISON  (n={len(pids)}, seed={seed})")
    print(f"{'='*70}")
    print(f"  {'Exp':<4} {'Label':<14} {'Mean AUC':>9} {'Med AUC':>9} "
          f"{'BalAcc':>8} {'F1':>8} {'Time':>6}")
    print(f"  {'-'*4} {'-'*14} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*6}")

    for ek in sorted(all_results.keys()):
        r = all_results[ek]
        print(f"  {ek:<4} {r['label']:<14} "
              f"{r['population_auc']['mean']:>9.4f} "
              f"{r['population_auc']['median']:>9.4f} "
              f"{r['population_balacc']['mean']:>8.4f} "
              f"{r['population_f1']['mean']:>8.4f} "
              f"{r['total_time_min']:>5.1f}m")

    print(f"\n  Total wall time: {t_grand / 60:.1f} min")
    print(f"  Results: {out_path}")


def _save(
    out_path: Path,
    all_results: dict,
    pids: list[str],
    feat_names: list[str],
    seed: int,
    elapsed_s: float,
) -> None:
    """Write current state to JSON (incremental save)."""
    doc = {
        "method":          "rbf_ablation_loso",
        "seed":            seed,
        "n_participants":  len(pids),
        "excluded":        sorted(_EXCLUDE),
        "n_features":      len(feat_names),
        "feature_names":   list(feat_names),
        "experiments":     all_results,
        "elapsed_min":     round(elapsed_s / 60, 1),
    }
    out_path.write_text(json.dumps(doc, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
