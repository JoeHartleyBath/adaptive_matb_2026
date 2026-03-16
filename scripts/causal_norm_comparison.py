"""Compare causal normalisation strategies for online-compatible z-scoring.

Session structure (per participant, temporal order):
  Fix0 (60 s) → Fix1 (60 s) → Forest0 (180 s) → Task0 (180 s)
  → Forest1 → Task1 → Forest2 → Task2 → Forest3 → Task3

Strategies
----------
none          No per-participant z-scoring (raw features).
pp_zscore     Non-causal baseline — z-score each participant's task features
              using the global mean/std of ALL their task epochs.  This is
              the current default in personalised_logreg.py (AUC ≈ 0.6725).
forest_only   Z-score each task block using the immediately preceding
              forest block's feature mean/std.
cumulative    Z-score each task block using feature stats from all
              fixation + forest blocks seen so far (causally).
fixation_only Z-score ALL task blocks using only fixation cross features
              (~120 s, the shortest possible calibration).
calibration   Z-score ALL task blocks using fixation + Forest0 feature
              stats (a fixed ~300 s early-session baseline).
expanding     Z-score each task block using all data from session start
              to the start of that block (fix + forests + prior tasks).

Evaluation
----------
28-fold LOSO LogisticRegression (C=0.001, balanced, StandardScaler in
pipeline).  Primary metric: mean AUC across held-out participants.

Usage
-----
    python scripts/causal_norm_comparison.py
    python scripts/causal_norm_comparison.py --strategies forest_only calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path setup — must precede local imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ml.pretrain_loader import PretrainDataDir  # noqa: E402
from eeg.extract_features import load_all_features  # noqa: E402
from compare_ml_models import (  # noqa: E402
    _build_region_map,
    _DATASET,
    _YAML_CONFIG,
    loso_cv,
)


def _load_exclude(cfg_path: Path) -> set[str]:
    """Load excluded PIDs from pretrain_qc.yaml."""
    cfg = yaml.safe_load(cfg_path.read_text())
    excluded = cfg.get("excluded_participants") or {}
    return set(excluded.keys())


_QC_CONFIG = _REPO_ROOT / "config" / "pretrain_qc.yaml"
_EXCLUDE = _load_exclude(_QC_CONFIG)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGIES = [
    "none",
    "pp_zscore",
    "forest_only",
    "cumulative",
    "fixation_only",
    "calibration",
    "expanding",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_zscore(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Z-score with safe denominator (replace near-zero std with 1)."""
    std_safe = std.copy()
    std_safe[std_safe < 1e-12] = 1.0
    return (X - mean) / std_safe


# ---------------------------------------------------------------------------
# Normalisation strategies
# ---------------------------------------------------------------------------

def apply_normalisation(
    data: dict[str, dict],
    strategy: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply *strategy* to every participant's features.

    Returns (X_by, y_by) ready for ``loso_cv()``.
    """
    X_by: dict[str, np.ndarray] = {}
    y_by: dict[str, np.ndarray] = {}

    for pid, d in data.items():
        task_X = d["task_X"].copy()
        task_y = d["task_y"]
        task_bidx = d["task_bidx"]
        forest_X = d["forest_X"]
        forest_bidx = d["forest_bidx"]
        fix_X = d["fix_X"]

        # ----------------------------------------------------------
        if strategy == "none":
            pass  # raw features, no normalisation

        # ----------------------------------------------------------
        elif strategy == "pp_zscore":
            # Non-causal: fit scaler on ALL task epochs (future included)
            sc = StandardScaler()
            task_X = sc.fit_transform(task_X)

        # ----------------------------------------------------------
        elif strategy == "forest_only":
            # Each task block i normalised by preceding Forest i
            for bidx in range(4):
                t_mask = task_bidx == bidx
                f_mask = forest_bidx == bidx
                if not t_mask.any() or not f_mask.any():
                    continue
                bl = forest_X[f_mask]
                task_X[t_mask] = _safe_zscore(
                    task_X[t_mask], bl.mean(axis=0), bl.std(axis=0)
                )

        # ----------------------------------------------------------
        elif strategy == "cumulative":
            # Task block i normalised by all fix + forest_0..i
            for bidx in range(4):
                t_mask = task_bidx == bidx
                if not t_mask.any():
                    continue
                parts: list[np.ndarray] = []
                if fix_X.shape[0] > 0:
                    parts.append(fix_X)
                for fi in range(bidx + 1):
                    fm = forest_bidx == fi
                    if fm.any():
                        parts.append(forest_X[fm])
                if not parts:
                    continue
                bl = np.concatenate(parts)
                task_X[t_mask] = _safe_zscore(
                    task_X[t_mask], bl.mean(axis=0), bl.std(axis=0)
                )

        # ----------------------------------------------------------
        elif strategy == "fixation_only":
            # Fixed baseline: fixation blocks only (~120 s)
            if fix_X.shape[0] > 0:
                task_X = _safe_zscore(
                    task_X, fix_X.mean(axis=0), fix_X.std(axis=0)
                )

        # ----------------------------------------------------------
        elif strategy == "calibration":
            # Fixed baseline: fixation + Forest0  (~300 s)
            parts: list[np.ndarray] = []
            if fix_X.shape[0] > 0:
                parts.append(fix_X)
            fm = forest_bidx == 0
            if fm.any():
                parts.append(forest_X[fm])
            if parts:
                bl = np.concatenate(parts)
                task_X = _safe_zscore(
                    task_X, bl.mean(axis=0), bl.std(axis=0)
                )

        # ----------------------------------------------------------
        elif strategy == "expanding":
            # Task block i normalised by everything before it:
            #   fix + forest_0..i + task_0..i-1
            for bidx in range(4):
                t_mask = task_bidx == bidx
                if not t_mask.any():
                    continue
                parts: list[np.ndarray] = []
                if fix_X.shape[0] > 0:
                    parts.append(fix_X)
                for fi in range(bidx + 1):
                    fm = forest_bidx == fi
                    if fm.any():
                        parts.append(forest_X[fm])
                # Include prior task blocks (original, un-normalised)
                for ti in range(bidx):
                    tm = task_bidx == ti
                    if tm.any():
                        parts.append(d["task_X"][tm])
                if not parts:
                    continue
                bl = np.concatenate(parts)
                task_X[t_mask] = _safe_zscore(
                    task_X[t_mask], bl.mean(axis=0), bl.std(axis=0)
                )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        X_by[pid] = task_X
        y_by[pid] = task_y

    return X_by, y_by


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=_DATASET)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
        help="Strategies to compare (default: all six)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: results/test_pretrain/causal_norm_comparison.json)",
    )
    args = parser.parse_args()

    # Setup
    data_dir = PretrainDataDir(args.dataset)
    ch_names = data_dir.channel_names()
    srate = data_dir.srate()
    region_map = _build_region_map(_YAML_CONFIG, ch_names)
    pids = sorted(p for p in data_dir.available_pids() if p not in _EXCLUDE)

    print(f"Dataset     : {args.dataset}")
    print(f"Participants: {len(pids)}")
    print(f"Strategies  : {', '.join(args.strategies)}")
    print(f"Seed        : {args.seed}")
    print()

    # ------------------------------------------------------------------
    # 1. Feature extraction (once for all strategies)
    # ------------------------------------------------------------------
    print("Extracting features for all block types ...")
    t_start = time.time()
    data = load_all_features(data_dir, pids, srate, region_map)
    feat_names = next(iter(data.values()))["feat_names"]
    print(f"Feature extraction complete  ({time.time() - t_start:.1f} s, "
          f"{len(feat_names)} features)\n")

    # ------------------------------------------------------------------
    # 2. Run LOSO for each strategy
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    for strategy in args.strategies:
        print(f"--- {strategy} ---")
        t0 = time.time()
        X_by, y_by = apply_normalisation(data, strategy)
        loso_res = loso_cv(X_by, y_by, model_name="LogReg", seed=args.seed)
        elapsed = time.time() - t0

        aucs = [v["auc"] for v in loso_res.values()]
        bals = [v["bal_acc"] for v in loso_res.values()]
        f1s = [v["f1"] for v in loso_res.values()]

        results[strategy] = dict(
            per_participant=loso_res,
            mean_auc=float(np.mean(aucs)),
            std_auc=float(np.std(aucs)),
            median_auc=float(np.median(aucs)),
            mean_bal_acc=float(np.mean(bals)),
            mean_f1=float(np.mean(f1s)),
            elapsed_s=round(elapsed, 1),
        )
        r = results[strategy]
        print(f"  AUC  {r['mean_auc']:.4f}  (std {r['std_auc']:.4f}, "
              f"median {r['median_auc']:.4f})  "
              f"BalAcc {r['mean_bal_acc']:.4f}  F1 {r['mean_f1']:.4f}  "
              f"[{elapsed:.1f} s]")
        print()

    # ------------------------------------------------------------------
    # 3. Comparison table
    # ------------------------------------------------------------------
    pp_auc = results.get("pp_zscore", {}).get("mean_auc", 0.6725)

    print("=" * 80)
    print(f"Causal normalisation comparison   (LOSO LogReg C=0.001, {len(pids)} participants)")
    print("=" * 80)
    print(
        f"{'Strategy':<15}  {'Mean AUC':>8}  {'Median':>7}  {'Std':>6}  "
        f"{'BalAcc':>6}  {'F1':>6}  {'Δ pp_zs':>8}  {'Online':>6}"
    )
    print("-" * 80)

    for strategy in args.strategies:
        r = results[strategy]
        delta = r["mean_auc"] - pp_auc
        online = "no" if strategy == "pp_zscore" else "yes"
        print(
            f"{strategy:<15}  {r['mean_auc']:>8.4f}  {r['median_auc']:>7.4f}  "
            f"{r['std_auc']:>6.4f}  {r['mean_bal_acc']:>6.4f}  {r['mean_f1']:>6.4f}  "
            f"{delta:>+8.4f}  {online:>6}"
        )

    print("=" * 80)

    # Best online-compatible
    online = [s for s in args.strategies if s != "pp_zscore"]
    if online:
        best = max(online, key=lambda s: results[s]["mean_auc"])
        gap = results[best]["mean_auc"] - pp_auc
        print(
            f"\nBest online-compatible: {best}  "
            f"(AUC {results[best]['mean_auc']:.4f}, "
            f"gap vs non-causal: {gap:+.4f})"
        )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_path = args.out or (
        _REPO_ROOT / "results" / "test_pretrain" / "causal_norm_comparison.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = dict(
        description="Causal normalisation strategy comparison",
        date="2026-03-12",
        seed=args.seed,
        model="LogReg",
        C=0.001,
        n_participants=len(pids),
        participants=pids,
        feature_names=feat_names,
        n_features=len(feat_names),
        strategies={s: results[s] for s in args.strategies},
    )
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
