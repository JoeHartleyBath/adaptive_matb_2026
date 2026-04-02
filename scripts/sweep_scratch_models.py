"""Scratch classifier sweep: train on MATB calibration data -> test on adaptation XDF.

Compares LogReg (reference), SVM linear, and SVM RBF with varied C and gamma
using the same scratch pipeline (outer LOW-norm -> SelectKBest k=30 ->
StandardScaler).

This uses the same train/test split as the original 4-way comparison:
  train on all cal XDFs -> evaluate on held-out adaptation XDF.

Metrics (matching evaluate_model_on_adaptation.py):
  - 3-class macro OvR AUC  (primary: same as original validation)
  - 3-class accuracy
  - HIGH vs NOT-HIGH binary AUC  (secondary: adaptation-policy relevance)
  - P(HIGH) mean by true level: LOW / MOD / HIGH

SVM investigation rationale:
  Initial sweep showed SVM-RBF AUC ~0.30, all P(H) collapsing to ~0.09
  regardless of true level.  Training diagnostics confirmed SVM *does* learn
  (84-96% train acc, separated per-class probs on training data), so the
  collapse is a generalisation failure: RBF draws tight boundaries that
  don't transfer to adaptation windows recorded ~30 min later (within-session
  EEG drift).
  Hypothesis: linear SVM (like LR) will be robust to this shift; the failure
  is kernel-specific, not a fundamental SVM limitation.

Conclusions (PSELF S001, 2026-03-27)
-------------------------------------
  Model              AUC3   AUCbin  P(H)|LOW  P(H)|HIGH
  LR_C0.003         0.638    0.713     0.542      0.796   <- deployed
  SVM_lin_C0.1      0.658    0.743     0.681      0.988
  SVM_rbf_C0.01_g0.010  0.686*  0.758  0.350    0.350   <- degenerate

  (*) AUC3=0.686 for RBF arises from an Acc=25% degenerate model (always
      predicts one class); P(H) flat across all levels — not usable.

  Hypothesis confirmed: SVM-linear generalises (AUCbin ~0.74), RBF fails
  because tight cal-block boundaries don't survive within-session EEG drift.

  LR_C0.003 retained as deployed model:
  - SVM-linear gains <0.03 AUCbin but P(H)|LOW jumps to 0.68+, which
    would saturate the adaptation scheduler at baseline
  - LR P(H) spread (LOW=0.54, HIGH=0.80) gives the clearest signal to the
    adaptation system

Usage
-----
    python scripts/sweep_scratch_models.py \\
        --xdf-dir C:/data/adaptive_matb/physiology/sub-PSELF/ses-S001/physio \\
        --adaptation-xdf C:/data/adaptive_matb/physiology/sub-PSELF/ses-S001/physio/sub-PSELF_ses-S001_task-matb_acq-adaptation_physio_old1.xdf \\
        --scenario experiment/scenarios/adaptive_automation_pself_c1_8min.txt \\
        --pid PSELF \\
        --offset 25.17
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
from calibrate_participant import _load_xdf_block  # noqa: E402
from eeg import EegPreprocessor, extract_windows, slice_block  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402

_DEFAULT_REGION_CFG = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"
_ANALYSIS_SRATE = 128.0
_K = 30
SEED = 42
_LABEL_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


def _build_candidates() -> list[tuple[str, Any]]:
    """Return (name, unfitted estimator) pairs to sweep."""
    candidates: list[tuple[str, Any]] = []

    # --- LogReg: 3 reference points around validated C=0.003 ---
    for c in [0.001, 0.003, 0.01]:
        candidates.append((
            f"LR_C{c}",
            LogisticRegression(
                C=c, solver="saga", l1_ratio=0.0, max_iter=2000,
                class_weight="balanced", random_state=SEED,
            ),
        ))

    # --- SVM Linear: test whether a linear boundary (like LR) generalises ---
    # If linear SVM works and RBF fails, the failure is kernel-specific.
    for c in [0.001, 0.01, 0.1, 1.0]:
        candidates.append((
            f"SVM_lin_C{c}",
            SVC(
                C=c, kernel="linear", probability=True,
                class_weight="balanced", random_state=SEED,
            ),
        ))

    # --- SVM RBF: vary C and gamma to probe overfitting sensitivity ---
    # gamma='scale' = 1/(n_feat * Var(X)); after StandardScaler Var=1 -> ~1/30 ~= 0.033
    # gamma=0.001 / 0.01 produce much smoother, less overfit boundaries
    for c in [0.01, 0.1, 1.0]:
        for gamma in ["scale", 0.001, 0.01]:
            g_str = gamma if isinstance(gamma, str) else f"{gamma:.3f}"
            candidates.append((
                f"SVM_rbf_C{c}_g{g_str}",
                SVC(
                    C=c, kernel="rbf", gamma=gamma, probability=True,
                    class_weight="balanced", random_state=SEED,
                ),
            ))

    return candidates


def _parse_adaptation_scenario(scenario_path: Path) -> list[tuple[float, float, str]]:
    """Parse a scenario .txt -> (start_s, end_s, level) per block."""
    _RE = re.compile(
        r"(?P<time>\d+:\d{2}:\d{2});labstreaminglayer;marker;"
        r"STUDY/V0/(?:adaptive_automation|calibration_condition)/\d+/block_\d+/(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
    )
    open_starts: dict[str, float] = {}
    blocks: list[tuple[float, float, str]] = []
    for line in scenario_path.read_text(encoding="utf-8").splitlines():
        m = _RE.match(line.strip().split("|")[0])
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


def _load_adaptation_features(
    xdf_path: Path,
    scenario_path: Path,
    offset_s: float,
    expected_channels: list[str],
    region_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load adaptation XDF and extract features with block-level labels.

    Returns (X, y) where y uses _LABEL_MAP (0=LOW, 1=MOD, 2=HIGH).
    """
    import pyxdf

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream in adaptation XDF.")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    scenario_blocks = _parse_adaptation_scenario(scenario_path)
    if not scenario_blocks:
        sys.exit(f"ERROR: No blocks parsed from {scenario_path}")

    matb_t0 = eeg_ts[0] + offset_s
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for start_s, end_s, level in scenario_blocks:
        start_idx = int(np.searchsorted(eeg_ts, matb_t0 + start_s))
        end_idx = int(np.searchsorted(eeg_ts, matb_t0 + end_s))
        block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] == 0:
            continue
        X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
        all_X.append(X)
        all_y.append(np.full(len(X), _LABEL_MAP[level], dtype=np.int64))

    if not all_X:
        sys.exit("ERROR: No windows extracted from adaptation XDF.")
    return np.concatenate(all_X), np.concatenate(all_y)


def _train_and_eval(
    cal_X_norm: np.ndarray,
    cal_y: np.ndarray,
    adapt_X_norm: np.ndarray,
    adapt_y: np.ndarray,
    estimator: Any,
    name: str = "",
    debug: bool = False,
) -> dict[str, float]:
    """Fit the scratch pipeline on cal data, evaluate on adaptation data.

    Pipeline: SelectKBest(k=30) -> StandardScaler -> estimator
    (outer normalisation is pre-applied: cal_X_norm / adapt_X_norm are
    already z-normalised with LOW-block stats, matching live calibration.)
    """
    k = min(_K, cal_X_norm.shape[1])
    selector = SelectKBest(f_classif, k=k)
    cal_X_sel = selector.fit_transform(cal_X_norm, cal_y)

    sc = StandardScaler()
    cal_X_sc = sc.fit_transform(cal_X_sel)

    clf = estimator.__class__(**estimator.get_params())
    clf.fit(cal_X_sc, cal_y)

    if debug and name:
        train_probs = clf.predict_proba(cal_X_sc)
        train_acc = float((np.argmax(train_probs, axis=1) == cal_y).mean())
        print(f"    classes_={list(clf.classes_)}  train_acc={train_acc:.1%}")
        for lbl, lvl in [(0, "LOW"), (1, "MOD"), (2, "HIGH")]:
            mask = cal_y == lbl
            if mask.any():
                print(f"    [TRAIN] P(col0,1,2)|{lvl}: "
                      f"{train_probs[mask, 0].mean():.3f}  "
                      f"{train_probs[mask, 1].mean():.3f}  "
                      f"{train_probs[mask, 2].mean():.3f}")

    adapt_X_sel = selector.transform(adapt_X_norm)
    adapt_X_sc = sc.transform(adapt_X_sel)
    probs = clf.predict_proba(adapt_X_sc)   # (N, n_classes) in classes_ order
    preds = clf.predict(adapt_X_sc)

    # classes_ is always sorted ascending: [0=LOW, 1=MOD, 2=HIGH]
    high_col = int(np.where(clf.classes_ == 2)[0][0])
    p_high = probs[:, high_col]
    is_high = (adapt_y == 2).astype(int)

    # 3-class macro OvR AUC (primary — matches original 4-way comparison)
    auc_macro = float(roc_auc_score(
        adapt_y, probs, multi_class="ovr", average="macro", labels=[0, 1, 2],
    ))
    # 3-class accuracy
    acc = float((preds == adapt_y).mean())
    # HIGH vs NOT-HIGH binary AUC (secondary — adaptation-policy relevance)
    auc_binary = float(roc_auc_score(is_high, p_high))

    level_p_high: dict[str, float] = {}
    for lbl, lvl in [(0, "LOW"), (1, "MOD"), (2, "HIGH")]:
        mask = adapt_y == lbl
        level_p_high[lvl] = float(p_high[mask].mean()) if mask.any() else float("nan")

    return {"auc_macro": auc_macro, "acc": acc, "auc_binary": auc_binary, **level_p_high}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scratch model sweep: train on cal data, test on adaptation XDF.")
    parser.add_argument("--xdf-dir", type=Path, required=True,
                        help="Directory with calibration .xdf files.")
    parser.add_argument("--adaptation-xdf", type=Path, required=True,
                        help="Held-out adaptation session .xdf file.")
    parser.add_argument("--scenario", type=Path, required=True,
                        help="Scenario .txt with block markers (relative to repo root or absolute).")
    parser.add_argument("--pid", type=str, required=True,
                        help="Participant ID (for display).")
    parser.add_argument("--offset", type=float, default=12.0,
                        help="Seconds from XDF start to scenario t=0 (default: 12.0).")
    parser.add_argument("--debug", action="store_true",
                        help="Print training-set P(H) diagnostics per model.")
    args = parser.parse_args()

    scenario_path = args.scenario if args.scenario.is_absolute() else _REPO_ROOT / args.scenario

    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    region_map = _build_region_map(_DEFAULT_REGION_CFG, expected_channels)

    # --- Load calibration XDFs ---
    xdf_dir = Path(args.xdf_dir)
    cal_xdfs = sorted(f for f in xdf_dir.glob("*.xdf") if "acq-cal_c" in f.name)
    if not cal_xdfs:
        cal_xdfs = sorted(xdf_dir.glob("*.xdf"))
    if not cal_xdfs:
        sys.exit(f"ERROR: No .xdf files found in {xdf_dir}")

    print(f"Participant  : {args.pid}")
    print(f"Cal XDF dir  : {xdf_dir}")
    print(f"Cal XDFs     : {[f.name for f in cal_xdfs]}")
    print(f"Adapt XDF    : {args.adaptation_xdf.name}")
    print(f"Scenario     : {scenario_path.name}")
    print(f"Offset       : {args.offset}s\n")

    blocks: list[tuple[np.ndarray, np.ndarray, str]] = []
    print("Loading calibration XDFs:")
    for xdf_path in cal_xdfs:
        result = _load_xdf_block(xdf_path, expected_channels)
        if result is None:
            continue
        for epochs, level in result:
            X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
            y = np.full(len(X), _LABEL_MAP[level], dtype=np.int64)
            blocks.append((X, y, level))

    if not blocks:
        sys.exit("ERROR: No blocks loaded from calibration XDFs.")

    cal_X_raw = np.concatenate([b[0] for b in blocks])
    cal_y = np.concatenate([b[1] for b in blocks])
    level_counts: dict[str, int] = {}
    for _, y, lvl in blocks:
        level_counts[lvl] = level_counts.get(lvl, 0) + len(y)

    print(f"  Blocks: {len(blocks)}  Windows: {len(cal_y)}  Per level: {level_counts}")

    # Outer normalisation using LOW blocks (matches live calibrate pipeline).
    low_X = np.concatenate([X for X, _, lvl in blocks if lvl == "LOW"])
    norm_mean = low_X.mean(axis=0)
    norm_std = low_X.std(axis=0)
    norm_std[norm_std < 1e-12] = 1.0
    cal_X_norm = (cal_X_raw - norm_mean) / norm_std

    # --- Load adaptation XDF ---
    print("\nLoading adaptation XDF ...")
    adapt_X_raw, adapt_y = _load_adaptation_features(
        args.adaptation_xdf, scenario_path, args.offset, expected_channels, region_map,
    )
    adapt_X_norm = (adapt_X_raw - norm_mean) / norm_std

    adapt_counts: dict[str, int] = {}
    for lbl, name in [(0, "LOW"), (1, "MOD"), (2, "HIGH")]:
        n = int((adapt_y == lbl).sum())
        if n:
            adapt_counts[name] = n
    print(f"  Windows: {len(adapt_y)}  Per level: {adapt_counts}\n")

    # --- Run sweep ---
    candidates = _build_candidates()

    hdr = (f"{'Model':<24}  {'AUC3':>6}  {'Acc':>6}  {'AUCbin':>7}"
           f"  {'P(H)|LOW':>8}  {'P(H)|MOD':>8}  {'P(H)|HIGH':>9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    results = []
    for name, estimator in candidates:
        print(f"  {name:<22} ...", end=" ", flush=True)
        m = _train_and_eval(cal_X_norm, cal_y, adapt_X_norm, adapt_y, estimator,
                            name=name, debug=args.debug)
        results.append((name, m))
        print(
            f"\r{name:<24}  "
            f"{m['auc_macro']:>6.3f}  {m['acc']:>6.1%}  {m['auc_binary']:>7.3f}"
            f"  {m['LOW']:>8.3f}  {m['MOD']:>8.3f}  {m['HIGH']:>9.3f}"
        )

    print(sep)
    best_name, best_m = max(results, key=lambda r: r[1]["auc_macro"])
    print(f"\nBest model   : {best_name}  (AUC3={best_m['auc_macro']:.3f}  Acc={best_m['acc']:.1%})")
    print(f"Current live : LR_C0.003")
    lr_003 = next((m for n, m in results if n == "LR_C0.003"), None)
    if lr_003:
        delta = best_m["auc_macro"] - lr_003["auc_macro"]
        print(f"Delta vs live: {delta:+.3f}  "
              f"({'consider switching' if delta > 0.02 else 'no meaningful improvement'})")
    print("\nDecision guide:")
    print("  AUC3    : 3-class macro OvR AUC (primary, matches original 4-way comparison)")
    print("  Acc     : 3-class accuracy")
    print("  AUCbin  : HIGH vs NOT-HIGH binary AUC (adaptation-policy relevance)")
    print("  P(H)|*  : want LOW << 0.5 << HIGH for adaptive automation thresholding")


if __name__ == "__main__":
    main()
