"""Sweep calibration block count to find the minimum needed for personalised MWL.

Each 9-min calibration XDF contains 9 × 1-min blocks in pseudorandomised level
order (3 repetitions of each: LOW / MODERATE / HIGH).  Grouping every 3
consecutive blocks yields a *triplet* — one 3-min unit that is guaranteed by the
study design to contain exactly 1 min of each level.

Two calibration XDFs → 6 triplets total.  This script sweeps every combination
of n triplets (n = 1 … 6) × a configurable set of model configs (LogReg,
SVM-linear, SVM-RBF, LDA, and RandomForest, each across a range of relevant
hyperparameters) and evaluates each model on a held-out 8-min adaptation XDF.

Note: gamma only applies to SVM-RBF; C only applies to LogReg/SVM;
n_estimators only applies to RF; LDA uses shrinkage='auto' (lsqr solver).

    AUC        — binary HIGH vs NOT-HIGH (ROC-AUC)
    Acc        — accuracy at threshold 0.5
    Youden's J — max(TPR − FPR) with the optimal threshold printed
    P(H)       — mean P(HIGH) per level (LOW / MODERATE / HIGH)

Norm stats are always derived from the resting-baseline XDF, matching the
behaviour of the real session.

A verification gate checks every triplet for exactly {LOW, MODERATE, HIGH}
before any feature extraction begins — the script aborts if the block structure
is unexpected.

Outputs
-------
    {out_csv}           — raw per-combination results (one row per model fit)
    {out_csv}_summary   — per-n summary (mean ± std across combinations)

Usage
-----
    python scripts/calibration_block_count_sweep.py \\
        --xdf-cal1  "C:/data/.../sub-PSELF_ses-S001_task-matb_acq-cal_c1_physio.xdf" \\
        --xdf-cal2  "C:/data/.../sub-PSELF_ses-S001_task-matb_acq-cal_c2_physio.xdf" \\
        --xdf-rest  "C:/data/.../sub-PSELF_ses-S001_task-matb_acq-rest_physio.xdf" \\
        --xdf-adapt "C:/data/.../sub-PSELF_ses-S001_task-matb_acq-adaptation_physio.xdf" \\
        --scenario-adapt experiment/scenarios/adaptive_automation_pself_c1_8min.txt \\
        --out-csv        results/cal_block_sweep_PSELF.csv \\
        --k-values       10 20 30 40 54 \\
        --logreg-c-values 0.003 \\
        --svm-c-values   0.01 0.1 1.0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyxdf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from build_mwl_training_dataset import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)
from calibrate_participant import (  # noqa: E402
    _ANALYSIS_SRATE,
    _DEFAULT_REGION_CFG,
    _LOGREG_C,
    _LOGREG_K,
    _compute_resting_norm,
    _load_rest_xdf_block,
    _load_xdf_block,
    SEED,
)
from eeg import EegPreprocessor, extract_windows, slice_block  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402

_LABEL_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
_LEVELS = ("LOW", "MODERATE", "HIGH")

# ---------------------------------------------------------------------------
# Adaptation scenario parser (mirrors evaluate_model_on_adaptation.py)
# ---------------------------------------------------------------------------

_ADAPT_RE = re.compile(
    r"(?P<time>\d+:\d{2}:\d{2});labstreaminglayer;marker;"
    r"STUDY/V0/(?:adaptive_automation|calibration_condition)/\d+/block_\d+/"
    r"(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
)


def _parse_adaptation_scenario(scenario_path: Path) -> list[tuple[float, float, str]]:
    """Parse adaptation scenario .txt → (start_s, end_s, level) per block."""
    open_starts: dict[str, float] = {}
    blocks: list[tuple[float, float, str]] = []
    for line in scenario_path.read_text(encoding="utf-8").splitlines():
        m = _ADAPT_RE.match(line.strip().split("|")[0])
        if not m:
            continue
        parts = m.group("time").split(":")
        t_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        level = m.group("level")
        if m.group("event") == "START":
            open_starts[level] = float(t_s)
        elif m.group("event") == "END" and level in open_starts:
            blocks.append((open_starts.pop(level), float(t_s), level))
    return blocks


# ---------------------------------------------------------------------------
# Build calibration triplets with verification gate
# ---------------------------------------------------------------------------

def _build_triplets(
    xdf_path: Path,
    expected_channels: list[str],
) -> list[list[tuple[np.ndarray, str]]]:
    """Load one calibration XDF and return chronological 3-block triplets.

    Each triplet is a list of 3 ``(epochs, level)`` pairs.  A verification
    gate checks that every triplet contains exactly one block of each level
    {LOW, MODERATE, HIGH} before returning — the function raises ValueError
    with a diagnostic message if this invariant is violated.

    Raises
    ------
    ValueError
        If the block count is not a multiple of 3, or if any triplet does not
        contain exactly one sub-block of each level.
    """
    blocks = _load_xdf_block(xdf_path, expected_channels)
    if blocks is None:
        raise ValueError(f"Failed to load calibration XDF: {xdf_path.name}")

    n = len(blocks)
    if n % 3 != 0:
        raise ValueError(
            f"{xdf_path.name}: expected a multiple-of-3 block count, got {n}.\n"
            f"  Levels found: {[lvl for _, lvl in blocks]}"
        )

    triplets = [blocks[i : i + 3] for i in range(0, n, 3)]

    for t_idx, triplet in enumerate(triplets):
        levels_found = {lvl for _, lvl in triplet}
        if levels_found != {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(
                f"{xdf_path.name}: triplet {t_idx + 1} does not contain one block "
                f"of each level.\n"
                f"  Expected: {{LOW, MODERATE, HIGH}}\n"
                f"  Found:    {{{', '.join(sorted(levels_found))}}}\n"
                f"  Block order in triplet: {[lvl for _, lvl in triplet]}"
            )

    return triplets


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_triplet_features(
    triplet: list[tuple[np.ndarray, str]],
    region_map: dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract and z-normalise features for one triplet (3 sub-blocks → (X, y))."""
    parts_X: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    for epochs, level in triplet:
        X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
        X_norm = (X - norm_mean) / norm_std
        y = np.full(len(X_norm), _LABEL_MAP[level], dtype=np.int64)
        parts_X.append(X_norm)
        parts_y.append(y)
    return np.concatenate(parts_X), np.concatenate(parts_y)


def _load_and_extract_adaptation(
    xdf_path: Path,
    scenario_path: Path,
    expected_channels: list[str],
    region_map: dict,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    offset_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load adaptation XDF → (X_adapt_norm, y_adapt_true, is_high).

    Mirrors the adaptation loading logic in evaluate_model_on_adaptation.py.
    """
    print(f"  Loading: {xdf_path.name} ...", flush=True)
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise ValueError(f"No EEG stream found in: {xdf_path.name}")

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        raise ValueError(
            f"Channel count mismatch in adaptation XDF: {n_ch} != {len(expected_channels)}"
        )

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"    Decimated {actual_srate:.0f} → {_ANALYSIS_SRATE:.0f} Hz (×{factor})")

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    scenario_blocks = _parse_adaptation_scenario(scenario_path)
    if not scenario_blocks:
        raise ValueError(f"No blocks parsed from scenario: {scenario_path}")

    matb_t0 = eeg_ts[0] + offset_s
    print(
        f"    EEG: {n_ch} ch, {len(eeg_ts)} samples, {eeg_ts[-1] - eeg_ts[0]:.1f}s"
        f"  |  scenario offset={offset_s}s"
    )
    print(f"    Block order: {' '.join(lvl[0] for _, _, lvl in scenario_blocks)}")

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for start_s, end_s, level in scenario_blocks:
        start_idx = int(np.searchsorted(eeg_ts, matb_t0 + start_s))
        end_idx = int(np.searchsorted(eeg_ts, matb_t0 + end_s))
        block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] == 0:
            print(f"    WARNING: {level} block yielded 0 windows — skipping")
            continue
        X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
        X_norm = (X - norm_mean) / norm_std
        all_X.append(X_norm)
        all_y.append(np.full(len(X_norm), _LABEL_MAP[level], dtype=np.int64))

    X_adapt = np.concatenate(all_X)
    y_adapt = np.concatenate(all_y)
    is_high = (y_adapt == _LABEL_MAP["HIGH"]).astype(np.int32)
    print(
        f"    Adaptation windows: {len(y_adapt)} total"
        f"  (LOW={int((y_adapt == 0).sum())}"
        f", MOD={int((y_adapt == 1).sum())}"
        f", HIGH={int((y_adapt == 2).sum())})"
    )
    return X_adapt, y_adapt, is_high


# ---------------------------------------------------------------------------
# Model fit + evaluate
# ---------------------------------------------------------------------------

def _fit_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    is_high: np.ndarray,
    model_cfg: dict,
) -> dict:
    """Fit one model config and evaluate on the adaptation data.

    model_cfg keys
    --------------
    model : "logreg" | "svm_lin" | "svm_rbf" | "lda" | "rf"
    k     : int         — SelectKBest top-k (f_classif)
    C     : float|nan   — regularisation strength (NaN for lda/rf)
    gamma : str|"na"    — "scale"/"auto" for svm_rbf; "na" for all others
    n_est : int         — n_estimators for rf; 0 for all others

    Returns AUC, accuracy at 0.5 threshold, Youden's J (optimal HIGH vs
    NOT-HIGH separator) with its threshold, and mean P(HIGH) per level.
    """
    k = min(model_cfg["k"], X_train.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X_train, y_train)

    if model_cfg["model"] == "logreg":
        clf = LogisticRegression(
            C=model_cfg["C"], solver="saga", l1_ratio=0.0, max_iter=2000,
            warm_start=False, class_weight="balanced", random_state=SEED,
        )
    elif model_cfg["model"] == "svm_lin":
        clf = SVC(
            kernel="linear", C=model_cfg["C"],
            class_weight="balanced", probability=True, random_state=SEED,
        )
    elif model_cfg["model"] == "svm_rbf":
        clf = SVC(
            kernel="rbf", C=model_cfg["C"], gamma=model_cfg["gamma"],
            class_weight="balanced", probability=True, random_state=SEED,
        )
    elif model_cfg["model"] == "lda":
        clf = LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr")
    else:  # rf
        clf = RandomForestClassifier(
            n_estimators=model_cfg["n_est"], class_weight="balanced_subsample",
            random_state=SEED, n_jobs=1,
        )

    pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
    pipe.fit(X_sel, y_train)

    X_adapt_sel = selector.transform(X_adapt)
    probs = pipe.predict_proba(X_adapt_sel)      # (N, n_classes)
    p_high = probs[:, probs.shape[1] - 1]        # P(HIGH) = last column

    auc = float(roc_auc_score(is_high, p_high))
    acc = float(((p_high >= 0.5) == is_high.astype(bool)).mean())
    fpr_arr, tpr_arr, thr_arr = roc_curve(is_high, p_high)
    j_scores = tpr_arr - fpr_arr
    best_j_idx = int(np.argmax(j_scores))
    youdens_j = float(j_scores[best_j_idx])
    j_threshold = float(thr_arr[best_j_idx])
    p_h_by_level = {
        f"p_high_{lvl.lower()}": float(p_high[y_adapt == _LABEL_MAP[lvl]].mean())
        if (y_adapt == _LABEL_MAP[lvl]).any() else float("nan")
        for lvl in _LEVELS
    }
    return {"auc": auc, "acc": acc, "youdens_j": youdens_j, "j_threshold": j_threshold, **p_h_by_level}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _default_out = (
        _REPO_ROOT / "results"
        / f"cal_block_sweep_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.csv"
    )
    parser = argparse.ArgumentParser(
        description="Sweep calibration block count vs adaptation model performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--xdf-cal1", type=Path, required=True,
                        help="First 9-min calibration XDF.")
    parser.add_argument("--xdf-cal2", type=Path, required=True,
                        help="Second 9-min calibration XDF.")
    parser.add_argument("--xdf-rest", type=Path, required=True,
                        help="Resting-baseline XDF (source of norm stats).")
    parser.add_argument("--xdf-adapt", type=Path, required=True,
                        help="Adaptation XDF (held-out evaluation).")
    parser.add_argument("--scenario-adapt", type=Path, required=True,
                        help="Adaptation scenario .txt (block timing).")
    parser.add_argument("--offset", type=float, default=12.0,
                        help="Seconds from XDF start to MATB t=0 (default: 12.0).")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 30, 40, 54],
                        help="SelectKBest k values to sweep (default: 10 20 30 40 54).")
    parser.add_argument("--logreg-c-values", type=float, nargs="+", default=[0.003],
                        help="LogReg C values to sweep (default: 0.003).")
    parser.add_argument("--svm-c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0],
                        help="SVM-linear C values to sweep (default: 0.01 0.1 1.0).")
    parser.add_argument("--svm-rbf-c-values", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                        help="SVM-RBF C values to sweep (default: 0.1 1.0 10.0).")
    parser.add_argument("--svm-rbf-gamma-values", type=str, nargs="+", default=["scale", "auto"],
                        help="SVM-RBF gamma values to sweep (default: scale auto).")
    parser.add_argument("--rf-n-estimators", type=int, nargs="+", default=[200],
                        help="RF n_estimators values to sweep (default: 200).")
    parser.add_argument("--out-csv", type=Path, default=_default_out,
                        help="Output CSV path (default: results/cal_block_sweep_{ts}.csv).")
    args = parser.parse_args()

    scenario_path = (
        args.scenario_adapt if args.scenario_adapt.is_absolute()
        else _REPO_ROOT / args.scenario_adapt
    )

    # Build model config list from CLI args
    model_configs: list[dict] = []
    for k in sorted(set(args.k_values)):
        for C in sorted(set(args.logreg_c_values)):
            model_configs.append({"model": "logreg",   "k": k, "C": C,            "gamma": "na", "n_est": 0})
        for C in sorted(set(args.svm_c_values)):
            model_configs.append({"model": "svm_lin",  "k": k, "C": C,            "gamma": "na", "n_est": 0})
        for C in sorted(set(args.svm_rbf_c_values)):
            for gamma in sorted(set(args.svm_rbf_gamma_values)):
                model_configs.append({"model": "svm_rbf", "k": k, "C": C,       "gamma": gamma, "n_est": 0})
        model_configs.append({"model": "lda",     "k": k, "C": float("nan"), "gamma": "na", "n_est": 0})
    for n_est in sorted(set(args.rf_n_estimators)):
        for k in sorted(set(args.k_values)):
            model_configs.append({"model": "rf",      "k": k, "C": float("nan"), "gamma": "na", "n_est": n_est})
    print(f"Model configs ({len(model_configs)} total):")
    for cfg in model_configs:
        extras = (f"  C={cfg['C']}" if not np.isnan(float(cfg["C"])) else "")
        extras += (f"  gamma={cfg['gamma']}" if cfg["gamma"] != "na" else "")
        extras += (f"  n_est={cfg['n_est']}" if cfg["n_est"] > 0 else "")
        print(f"  {cfg['model']:<9}  k={cfg['k']:>3}{extras}")

    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    region_map = _build_region_map(_DEFAULT_REGION_CFG, expected_channels)

    # ---- Phase 1: Resting baseline → norm stats ----
    print("\n=== Phase 1: Resting baseline ===")
    rest_epochs = _load_rest_xdf_block(args.xdf_rest, expected_channels)
    if rest_epochs is None:
        sys.exit("ERROR: Failed to load resting-baseline XDF.")
    norm_mean, norm_std = _compute_resting_norm(rest_epochs, _ANALYSIS_SRATE, region_map)
    print(f"  Norm stats computed from {rest_epochs.shape[0]} rest windows.")

    # ---- Phase 2: Calibration XDFs → 6 verified triplets ----
    print("\n=== Phase 2: Calibration XDFs ===")
    all_triplets: list[list[tuple[np.ndarray, str]]] = []
    for xdf_idx, xdf_path in enumerate([args.xdf_cal1, args.xdf_cal2], start=1):
        print(f"  Cal XDF {xdf_idx}: {xdf_path.name}")
        triplets = _build_triplets(xdf_path, expected_channels)
        for t_idx, triplet in enumerate(triplets):
            levels_in_triplet = [lvl for _, lvl in triplet]
            print(f"    Triplet {t_idx + 1}: {levels_in_triplet}  [verified]")
        all_triplets.extend(triplets)

    n_triplets = len(all_triplets)
    if n_triplets != 6:
        sys.exit(f"ERROR: Expected 6 triplets total, got {n_triplets}.")
    print(f"\n  All {n_triplets} triplets verified — each contains 1×LOW, 1×MODERATE, 1×HIGH.")

    # ---- Phase 3: Pre-extract calibration features (done once, not per combo) ----
    print("\n=== Phase 3: Pre-extracting calibration features ===")
    triplet_data: list[tuple[np.ndarray, np.ndarray]] = []
    for t_idx, triplet in enumerate(all_triplets):
        src_xdf = 1 if t_idx < 3 else 2
        print(f"  Triplet {t_idx + 1}/6 (XDF {src_xdf}) ...", end=" ", flush=True)
        X, y = _extract_triplet_features(triplet, region_map, norm_mean, norm_std)
        print(
            f"{len(y)} windows"
            f"  (LOW={int((y == 0).sum())}"
            f", MOD={int((y == 1).sum())}"
            f", HIGH={int((y == 2).sum())})"
        )
        triplet_data.append((X, y))

    # ---- Phase 4: Adaptation features ----
    print("\n=== Phase 4: Adaptation XDF ===")
    X_adapt, y_adapt, is_high = _load_and_extract_adaptation(
        args.xdf_adapt, scenario_path, expected_channels,
        region_map, norm_mean, norm_std, args.offset,
    )

    # ---- Phase 5: Sweep ----
    n_combos = sum(
        len(list(itertools.combinations(range(n_triplets), n)))
        for n in range(1, n_triplets + 1)
    )
    total_fits = n_combos * len(model_configs)
    print(
        f"\n=== Phase 5: Sweep (n=1..{n_triplets}, "
        f"{len(model_configs)} model configs, {total_fits} total fits) ==="
    )

    raw_rows: list[dict] = []
    fit_count = 0
    for n in range(1, n_triplets + 1):
        for combo in itertools.combinations(range(n_triplets), n):
            X_train = np.concatenate([triplet_data[i][0] for i in combo])
            y_train = np.concatenate([triplet_data[i][1] for i in combo])
            combo_str = "_".join(str(i + 1) for i in combo)
            for model_cfg in model_configs:
                fit_count += 1
                metrics = _fit_and_evaluate(
                    X_train, y_train, X_adapt, y_adapt, is_high, model_cfg
                )
                _tag = model_cfg["model"] + f"_k{model_cfg['k']}"
                if not np.isnan(float(model_cfg["C"])):
                    _tag += f"_C{model_cfg['C']}"
                if model_cfg["gamma"] != "na":
                    _tag += f"_g{model_cfg['gamma']}"
                if model_cfg["n_est"] > 0:
                    _tag += f"_n{model_cfg['n_est']}"
                raw_rows.append({
                    "n_blocks": n,
                    "combo": combo_str,
                    "model": model_cfg["model"],
                    "k": model_cfg["k"],
                    "C": model_cfg["C"],
                    "gamma": model_cfg["gamma"],
                    "n_est": model_cfg["n_est"],
                    "n_train_windows": len(y_train),
                    **metrics,
                })
                print(
                    f"  [{fit_count:>4}/{total_fits}]"
                    f"  n={n}  combo={combo_str:<11}  {_tag:<32}"
                    f"  AUC={metrics['auc']:.3f}"
                    f"  Acc={metrics['acc']:.1%}"
                    f"  J={metrics['youdens_j']:.3f}"
                    f"  thr={metrics['j_threshold']:.3f}"
                    f"  P(H)|L={metrics['p_high_low']:.3f}"
                    f"  P(H)|H={metrics['p_high_high']:.3f}",
                    flush=True,
                )

    # ---- Phase 6: Write raw CSV ----
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"\n  Raw results ({len(raw_rows)} rows): {args.out_csv}")

    # ---- Phase 7: Summary table per model config (n=1..6 trend) + CSV ----
    # Group by (model, k, C); within each group show AUC mean±std across combos per n.
    cfg_keys = sorted(
        {(r["model"], r["k"], str(r["C"]), r["gamma"], r["n_est"]) for r in raw_rows},
        key=lambda x: (x[0], x[1], x[2], x[3], x[4]),
    )

    summary_rows: list[dict] = []
    for model, k, C_str, gamma, n_est in cfg_keys:
        cfg_subset = [
            r for r in raw_rows
            if r["model"] == model and r["k"] == k
            and str(r["C"]) == C_str and r["gamma"] == gamma and r["n_est"] == n_est
        ]
        cfg_desc = f"{model}  k={k}"
        if C_str != "nan":
            cfg_desc += f"  C={C_str}"
        if gamma != "na":
            cfg_desc += f"  gamma={gamma}"
        if n_est > 0:
            cfg_desc += f"  n_est={n_est}"
        print(f"\n  {cfg_desc}")
        print(
            f"  {'n':>4}  {'combos':>6}  {'AUC mean':>9}  {'AUC std':>8}"
            f"  {'J mean':>8}  {'J std':>7}  {'thr':>7}"
            f"  {'P(H)|L':>8}  {'P(H)|M':>8}  {'P(H)|H':>8}"
        )
        print("  " + "-" * 88)
        for n in range(1, n_triplets + 1):
            subset = [r for r in cfg_subset if r["n_blocks"] == n]
            if not subset:
                continue
            aucs = np.array([r["auc"] for r in subset])
            accs = np.array([r["acc"] for r in subset])
            js   = np.array([r["youdens_j"] for r in subset])
            jthr = np.array([r["j_threshold"] for r in subset])
            ph_l = np.array([r["p_high_low"] for r in subset])
            ph_m = np.array([r["p_high_moderate"] for r in subset])
            ph_h = np.array([r["p_high_high"] for r in subset])
            print(
                f"  {n:>4}  {len(subset):>6}  {aucs.mean():>9.3f}  {aucs.std():>8.3f}"
                f"  {js.mean():>8.3f}  {js.std():>7.3f}  {jthr.mean():>7.3f}"
                f"  {ph_l.mean():>8.3f}  {ph_m.mean():>8.3f}  {ph_h.mean():>8.3f}"
            )
            summary_rows.append({
                "model": model,
                "k": k,
                "C": C_str,
                "gamma": gamma,
                "n_est": n_est,
                "n_blocks": n,
                "n_combos": len(subset),
                "auc_mean": round(float(aucs.mean()), 4),
                "auc_std": round(float(aucs.std()), 4),
                "acc_mean": round(float(accs.mean()), 4),
                "acc_std": round(float(accs.std()), 4),
                "j_mean": round(float(js.mean()), 4),
                "j_std": round(float(js.std()), 4),
                "j_threshold_mean": round(float(jthr.mean()), 4),
                "p_high_low_mean": round(float(ph_l.mean()), 4),
                "p_high_moderate_mean": round(float(ph_m.mean()), 4),
                "p_high_high_mean": round(float(ph_h.mean()), 4),
            })

    summary_path = args.out_csv.with_stem(args.out_csv.stem + "_summary")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  Summary ({len(summary_rows)} rows): {summary_path}")


if __name__ == "__main__":
    main()
