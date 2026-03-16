"""Optimise LogReg hyperparameters for cross-subject MWL classification.

Two modes:

  K sweep (default):
    Sweeps SelectKBest(f_classif, k) across K values and reports
    mean AUC +/- std over multiple seeds.

  Joint sweep (--sweep):
    Sweeps K x C x penalty x scaler x window_s x step_s jointly with
    joblib parallelisation.  Window/overlap configurations are evaluated
    by re-extracting features from continuous HDF5 data (per-config
    caches avoid redundant extraction).
    Reports ranked configs and the winning feature set.

SelectKBest is fitted on training participants only inside each holdout
split, so there is no information leakage.

Usage (from MATB repo root):
    # K-only sweep (default, 10 seeds)
    .venv\\Scripts\\python.exe scripts/optimise_logreg.py

    # Joint hyperparameter sweep (4 workers)
    .venv\\Scripts\\python.exe scripts/optimise_logreg.py --sweep --n-seeds 10

    # More seeds / custom output
    .venv\\Scripts\\python.exe scripts/optimise_logreg.py --sweep --n-seeds 20 --out results/test_pretrain/joint_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from eeg.eeg_windower import WindowConfig  # noqa: E402
from ml.pretrain_loader import (  # noqa: E402
    PretrainDataDir,
    calibration_norm_features,
    load_baseline_from_cache,
)
from eeg.extract_features import load_all_features, _build_region_map  # noqa: E402
from compare_ml_models import (  # noqa: E402
    _DATASET,
    _YAML_CONFIG,
)

# ---------------------------------------------------------------------------
# Paths & constants (shared with compare_ml_models)
# ---------------------------------------------------------------------------

_NORM_CACHE = _REPO_ROOT / "results" / "test_pretrain" / "norm_comparison_features.npz"
_QC_CONFIG  = _REPO_ROOT / "config" / "pretrain_qc.yaml"

_DEFAULT_C = 0.001  # tuned via C sweep in compare_ml_models.py

_K_SWEEP_VALUES = [10, 15, 20, 25, 30, 35, 40, 45, 54]

# Joint sweep grid (Phase 2) — K narrowed to peak region from Phase 1
_SWEEP_K = [20, 25, 30, 35]
_SWEEP_C = [0.001, 0.003, 0.01, 0.03, 0.1]
_SWEEP_PENALTY = ["l2", "elasticnet"]
_SWEEP_SCALER  = ["standard", "robust"]
_ELASTICNET_L1_RATIO = 0.5
_N_WORKERS = 4

# Window / overlap sweep grid
_SWEEP_WINDOW = [1.0, 2.0, 4.0]   # window_s in seconds
_SWEEP_STEP   = [0.25, 0.5, 1.0]  # step_s in seconds (overlap = 1 - step/window)


def _load_exclude(cfg_path: Path) -> set[str]:
    cfg = yaml.safe_load(cfg_path.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_EXCLUDE = _load_exclude(_QC_CONFIG)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auc(y: np.ndarray, probs: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        return 0.5


def _make_logreg_pipeline(
    seed: int,
    k: int | None = None,
    C: float = _DEFAULT_C,
    penalty: str = "l2",
    scaler: str = "standard",
) -> Pipeline:
    """LogReg pipeline with configurable scaler, feature selection, and penalty."""
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    steps: list[tuple] = [("sc", sc)]
    if k is not None:
        steps.append(("sel", SelectKBest(f_classif, k=k)))
    clf_kwargs: dict = dict(
        C=C, max_iter=2000, class_weight="balanced", random_state=seed,
    )
    if penalty == "elasticnet":
        clf_kwargs["solver"] = "saga"
        clf_kwargs["l1_ratio"] = _ELASTICNET_L1_RATIO
    else:
        clf_kwargs["l1_ratio"] = 0  # l1_ratio=0 is equivalent to L2
    steps.append(("clf", LogisticRegression(**clf_kwargs)))
    return Pipeline(steps)


def holdout_k(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    k: int | None,
    test_frac: float,
    seed: int,
    C: float = _DEFAULT_C,
    penalty: str = "l2",
    scaler: str = "standard",
) -> dict:
    """Cross-subject holdout with optional feature selection.

    k=None or k >= n_features → no selection (baseline).
    """
    rng = np.random.default_rng(seed)
    pids = sorted(X_by.keys())
    n_features = next(iter(X_by.values())).shape[1]

    # Treat k >= n_features as "no selection"
    effective_k = k if (k is not None and k < n_features) else None

    n_test = max(1, round(len(pids) * test_frac))
    test_pids = list(rng.choice(pids, size=n_test, replace=False))
    train_pids = [p for p in pids if p not in test_pids]

    X_train = np.concatenate([X_by[p] for p in train_pids])
    y_train = np.concatenate([y_by[p] for p in train_pids])

    pipe = _make_logreg_pipeline(seed, k=effective_k, C=C,
                                  penalty=penalty, scaler=scaler)
    pipe.fit(X_train, y_train)

    per_pid: dict[str, float] = {}
    all_true, all_prob = [], []
    for pid in test_pids:
        probs = pipe.predict_proba(X_by[pid])[:, 1]
        per_pid[pid] = _auc(y_by[pid], probs)
        all_true.append(y_by[pid])
        all_prob.append(probs)

    return {
        "overall_auc": _auc(np.concatenate(all_true), np.concatenate(all_prob)),
        "per_pid": per_pid,
        "test_pids": test_pids,
    }


def _eval_one(
    X_by: dict[str, np.ndarray],
    y_by: dict[str, np.ndarray],
    k: int, C: float, penalty: str, scaler: str,
    test_frac: float, seed: int,
) -> float:
    """Single holdout evaluation — returns overall AUC.  Used by joblib."""
    r = holdout_k(X_by, y_by, k=k, test_frac=test_frac, seed=seed,
                  C=C, penalty=penalty, scaler=scaler)
    return r["overall_auc"]


# ---------------------------------------------------------------------------
# Data loading (from norm cache — same as compare_ml_models.py)
# ---------------------------------------------------------------------------

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
            f"       Run causal_norm_comparison.py first to build it."
        )

    npz = np.load(_NORM_CACHE, allow_pickle=False)
    npz_pids = list(npz["pids"])
    available = [p for p in npz_pids if p not in _EXCLUDE]
    if not available:
        raise SystemExit("No participants remain after exclusion.")

    feat_names = list(npz["feat_names"])
    X_by_raw: dict[str, np.ndarray] = {}
    y_by: dict[str, np.ndarray] = {}
    # Stress labels may be absent in caches built before this feature
    has_stress = f"{available[0]}_task_stress_y" in npz
    stress_y_by: dict[str, np.ndarray] | None = {} if has_stress else None
    for pid in available:
        X_by_raw[pid] = npz[f"{pid}_task_X"]
        y_by[pid] = npz[f"{pid}_task_y"]
        if has_stress:
            stress_y_by[pid] = npz[f"{pid}_task_stress_y"]  # type: ignore[index]

    # Calibration normalisation (ADR-0004)
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


def _load_data_for_window(
    window_s: float,
    step_s: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray],
           dict[str, np.ndarray] | None, list[str], list[str]]:
    """Extract features at a specific window/step config with per-config cache.

    Returns (X_by, y_by, stress_y_by, feat_names, pids) — same shape as
    load_data() but with features extracted at the given window config.
    """
    win = WindowConfig(window_s=window_s, step_s=step_s, srate=128.0)
    data_dir = PretrainDataDir(_DATASET, win=win)
    srate = data_dir.srate()
    ch_names = data_dir.channel_names()
    region_map = _build_region_map(_YAML_CONFIG, ch_names)
    pids_all = sorted(p for p in data_dir.available_pids() if p not in _EXCLUDE)

    win_tag = f"w{window_s}_s{step_s}"
    cache_path = (_REPO_ROOT / "results" / "test_pretrain"
                  / f"norm_{win_tag}.npz")

    data = load_all_features(
        data_dir, pids_all, srate, region_map,
        cache_path=cache_path, win_tag=win_tag,
    )

    feat_names = next(iter(data.values()))["feat_names"]
    X_by_raw: dict[str, np.ndarray] = {}
    y_by: dict[str, np.ndarray] = {}
    stress_y_by: dict[str, np.ndarray] | None = {}
    for pid in pids_all:
        d = data[pid]
        X_by_raw[pid] = d["task_X"]
        y_by[pid] = d["task_y"]
        if d.get("task_stress_y") is not None:
            stress_y_by[pid] = d["task_stress_y"]  # type: ignore[index]
        else:
            stress_y_by = None

    # Calibration normalisation (ADR-0004)
    baseline_by: dict[str, dict[str, np.ndarray]] = {}
    for pid in pids_all:
        d = data[pid]
        baseline_by[pid] = {
            "fix_X": d["fix_X"],
            "forest_X": d["forest_X"],
            "forest_bidx": d["forest_bidx"],
        }

    X_by: dict[str, np.ndarray] = {}
    for pid in pids_all:
        bl = baseline_by[pid]
        X_by[pid] = calibration_norm_features(
            X_by_raw[pid], bl["fix_X"], bl["forest_X"], bl["forest_bidx"],
        )

    return X_by, y_by, stress_y_by, feat_names, sorted(pids_all)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--n-seeds",    type=int,   default=10)
    ap.add_argument("--test-frac",  type=float, default=0.30)
    ap.add_argument("--out",        type=Path,  default=None,
                    help="Save JSON results to this path.")
    ap.add_argument("--sweep",      action="store_true",
                    help="Joint K x C x penalty x scaler x window x step sweep.")
    ap.add_argument("--stress",     action="store_true",
                    help="Classify stress (High/Low) instead of cognitive load.")
    ap.add_argument("--n-workers",  type=int,  default=_N_WORKERS,
                    help=f"Joblib workers for --sweep (default: {_N_WORKERS}).")
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.n_seeds)]

    # ==================================================================
    # Joint sweep (--sweep) — K x C x penalty x scaler x window x step
    # ==================================================================
    if args.sweep:
        target_name = "stress" if args.stress else "cognitive_load"
        print(f"Target       : {target_name}")
        print(f"Seeds        : {seeds[0]}..{seeds[-1]} ({args.n_seeds} splits)")
        print(f"Test frac    : {args.test_frac:.0%}")
        print()

        win_configs = list(product(_SWEEP_WINDOW, _SWEEP_STEP))
        hp_configs = list(product(_SWEEP_K, _SWEEP_C, _SWEEP_PENALTY, _SWEEP_SCALER))
        n_hp = len(hp_configs)
        n_win = len(win_configs)
        n_total = n_win * n_hp * args.n_seeds
        print(f"Joint sweep: {n_win} window configs x {n_hp} HP configs "
              f"x {args.n_seeds} seeds = {n_total} fits  "
              f"({args.n_workers} workers)")
        print()

        # Accumulate stats across all window configs
        config_stats: list[dict] = []

        for wi, (win_s, step_s) in enumerate(win_configs):
            print(f"[Window {wi + 1}/{n_win}]  "
                  f"window_s={win_s}  step_s={step_s}")
            X_by_w, y_by_cog_w, stress_y_by_w, feat_names_w, pids_w = \
                _load_data_for_window(win_s, step_s)

            if args.stress:
                if stress_y_by_w is None:
                    raise SystemExit(
                        "ERROR: Stress labels not found.\n"
                        "       Rebuild cache: python scripts/causal_norm_comparison.py"
                    )
                y_by_w = stress_y_by_w
            else:
                y_by_w = y_by_cog_w

            # Build flat job list for this window config
            jobs = [
                (ci, s, k, C, pen, sc)
                for ci, (k, C, pen, sc) in enumerate(hp_configs)
                for s in seeds
            ]

            results_flat = Parallel(n_jobs=args.n_workers, verbose=5)(
                delayed(_eval_one)(
                    X_by_w, y_by_w, k, C, pen, sc, args.test_frac, s,
                )
                for (_, s, k, C, pen, sc) in jobs
            )

            # Group results by HP config index
            grid_results: dict[int, list[float]] = {
                ci: [] for ci in range(n_hp)
            }
            for (ci, *_rest), auc_val in zip(jobs, results_flat):
                grid_results[ci].append(auc_val)

            # Compute stats per HP config for this window
            for ci, (k, C, pen, sc) in enumerate(hp_configs):
                aucs = grid_results[ci]
                config_stats.append({
                    "window_s": win_s, "step_s": step_s,
                    "k": k, "C": C, "penalty": pen, "scaler": sc,
                    "mean_auc":   float(np.mean(aucs)),
                    "std_auc":    float(np.std(aucs)),
                    "median_auc": float(np.median(aucs)),
                    "per_seed":   aucs,
                })
            print()

        # Sort by mean AUC descending
        config_stats.sort(key=lambda x: x["mean_auc"], reverse=True)

        # Print top 15
        print(f"\n{'Rank':>4}  {'Win':>4}  {'Step':>5}  {'K':>3}  {'C':>7}  "
              f"{'Penalty':<11}  {'Scaler':<9}  {'Mean AUC':>9}  "
              f"{'Std':>6}  {'Median':>7}")
        print("-" * 90)
        for i, cs in enumerate(config_stats[:15], 1):
            print(f"{i:>4}  {cs['window_s']:>4.1f}  {cs['step_s']:>5.2f}  "
                  f"{cs['k']:>3}  {cs['C']:>7.3f}  {cs['penalty']:<11}  "
                  f"{cs['scaler']:<9}  {cs['mean_auc']:>9.4f}  "
                  f"{cs['std_auc']:>6.4f}  {cs['median_auc']:>7.4f}")

        # Baseline comparison (default window, K=35, C=0.001, l2, standard)
        baseline_cfg = next(
            (cs for cs in config_stats
             if cs["window_s"] == 2.0 and cs["step_s"] == 0.5
             and cs["k"] == max(_SWEEP_K) and cs["C"] == _DEFAULT_C
             and cs["penalty"] == "l2" and cs["scaler"] == "standard"),
            None,
        )
        winner = config_stats[0]
        print()
        print(f"Winner : win={winner['window_s']}s  step={winner['step_s']}s  "
              f"K={winner['k']}  C={winner['C']}  "
              f"{winner['penalty']}  {winner['scaler']}  "
              f"-> {winner['mean_auc']:.4f} +/- {winner['std_auc']:.4f}")
        if baseline_cfg:
            delta = winner["mean_auc"] - baseline_cfg["mean_auc"]
            print(f"Baseline (w=2.0 s=0.5 K={baseline_cfg['k']} "
                  f"C={_DEFAULT_C} l2 standard): "
                  f"{baseline_cfg['mean_auc']:.4f}  delta = {delta:+.4f}")

        # Feature ranking at winning config's window
        best_k = winner["k"]
        X_by_best, y_by_best, stress_y_best, feat_names_best, pids_best = \
            _load_data_for_window(winner["window_s"], winner["step_s"])
        if args.stress and stress_y_best is not None:
            y_by_best = stress_y_best
        X_all = np.concatenate([X_by_best[p] for p in pids_best])
        y_all = np.concatenate([y_by_best[p] for p in pids_best])
        f_scores, p_values = f_classif(X_all, y_all)
        ranked_idx = np.argsort(f_scores)[::-1]
        selected_set = set(ranked_idx[:best_k].tolist())

        print(f"\n{'Rank':>4}  {'Feature':<22}  {'F-score':>8}  {'p-value':>10}  "
              f"{'Selected':>8}")
        print("-" * 60)
        for rank, idx in enumerate(ranked_idx, 1):
            mark = "  Y" if idx in selected_set else ""
            print(f"{rank:>4}  {feat_names_best[idx]:<22}  "
                  f"{f_scores[idx]:>8.1f}  "
                  f"{p_values[idx]:>10.2e}  {mark:>8}")

        # Save JSON
        n_features_best = len(feat_names_best)
        out_data = {
            "mode":           "joint_sweep",
            "seed":           args.seed,
            "n_seeds":        args.n_seeds,
            "test_frac":      args.test_frac,
            "n_participants": len(pids_best),
            "n_features":     n_features_best,
            "grid": {
                "K": _SWEEP_K, "C": _SWEEP_C,
                "penalty": _SWEEP_PENALTY, "scaler": _SWEEP_SCALER,
                "window_s": _SWEEP_WINDOW, "step_s": _SWEEP_STEP,
                "elasticnet_l1_ratio": _ELASTICNET_L1_RATIO,
            },
            "n_configs":      len(config_stats),
            "configs":        config_stats,
            "best": winner,
            "feature_ranking": [
                {
                    "rank": rank, "feature": feat_names_best[idx],
                    "f_score": float(f_scores[idx]),
                    "p_value": float(p_values[idx]),
                    "selected": int(idx) in selected_set,
                }
                for rank, idx in enumerate(ranked_idx, 1)
            ],
        }
        if args.out is not None:
            out_path = args.out.expanduser().resolve()
        else:
            out_path = _REPO_ROOT / "results" / "test_pretrain" / "logreg_joint_sweep.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved: {out_path}")
        return

    # ==================================================================
    # K-only sweep (default mode)
    # ==================================================================
    X_by, y_by_cog, stress_y_by, feat_names, pids = load_data()
    n_features = len(feat_names)

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

    print(f"Target       : {target_name}")
    print(f"Participants : {len(pids)}")
    print(f"Features     : {n_features}")
    print(f"Seeds        : {seeds[0]}..{seeds[-1]} ({args.n_seeds} splits)")
    print(f"Test frac    : {args.test_frac:.0%}")
    print()

    k_values = [k for k in _K_SWEEP_VALUES if k <= n_features]
    if n_features not in k_values:
        k_values.append(n_features)

    print(f"{'K':>4}  {'Mean AUC':>9}  {'Std':>6}  {'Median':>7}  "
          f"{'Min':>6}  {'Max':>6}  {'Δ baseline':>10}")
    print("-" * 62)

    sweep_results: dict[int, dict] = {}
    baseline_auc: float | None = None

    for k in k_values:
        seed_aucs = []
        for s in seeds:
            r = holdout_k(X_by, y_by, k=k, test_frac=args.test_frac, seed=s)
            seed_aucs.append(r["overall_auc"])

        mean_auc = float(np.mean(seed_aucs))
        std_auc  = float(np.std(seed_aucs))
        med_auc  = float(np.median(seed_aucs))

        if k == n_features:
            baseline_auc = mean_auc

        delta = (mean_auc - baseline_auc) if baseline_auc is not None else 0.0
        delta_str = f"{delta:+.4f}" if baseline_auc is not None else "  base"

        print(f"{k:>4}  {mean_auc:>9.4f}  {std_auc:>6.4f}  {med_auc:>7.4f}  "
              f"{min(seed_aucs):>6.4f}  {max(seed_aucs):>6.4f}  {delta_str:>10}")

        sweep_results[k] = {
            "mean_auc":   mean_auc,
            "std_auc":    std_auc,
            "median_auc": med_auc,
            "per_seed":   seed_aucs,
        }

    # ------------------------------------------------------------------
    # Best K
    # ------------------------------------------------------------------
    best_k = max(sweep_results, key=lambda k: sweep_results[k]["mean_auc"])
    best = sweep_results[best_k]
    print()
    print(f"Best K = {best_k}  (mean AUC {best['mean_auc']:.4f} "
          f"± {best['std_auc']:.4f})")

    # ------------------------------------------------------------------
    # Global feature ranking (f_classif on pooled data)
    # ------------------------------------------------------------------
    X_all = np.concatenate([X_by[p] for p in pids])
    y_all = np.concatenate([y_by[p] for p in pids])
    f_scores, p_values = f_classif(X_all, y_all)

    ranked_idx = np.argsort(f_scores)[::-1]

    print()
    print(f"{'Rank':>4}  {'Feature':<22}  {'F-score':>8}  {'p-value':>10}  "
          f"{'In top-{best_k}':>10}")
    print("-" * 62)
    selected_set = set(ranked_idx[:best_k])
    for rank, idx in enumerate(ranked_idx, 1):
        mark = "  ✓" if idx in selected_set else ""
        print(f"{rank:>4}  {feat_names[idx]:<22}  {f_scores[idx]:>8.1f}  "
              f"{p_values[idx]:>10.2e}  {mark:>10}")

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_data = {
        "seed":           args.seed,
        "n_seeds":        args.n_seeds,
        "test_frac":      args.test_frac,
        "C":              _DEFAULT_C,
        "n_participants": len(pids),
        "n_features":     n_features,
        "feature_names":  feat_names,
        "k_sweep":        {str(k): v for k, v in sweep_results.items()},
        "best_k":         best_k,
        "feature_ranking": [
            {
                "rank":     rank,
                "feature":  feat_names[idx],
                "f_score":  float(f_scores[idx]),
                "p_value":  float(p_values[idx]),
                "selected": int(idx) in selected_set,
            }
            for rank, idx in enumerate(ranked_idx, 1)
        ],
    }

    if args.out is not None:
        out_path = args.out.expanduser().resolve()
    else:
        out_path = _REPO_ROOT / "results" / "test_pretrain" / "logreg_k_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
