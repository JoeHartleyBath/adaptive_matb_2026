"""Evaluate the personalised MWL model on a held-out adaptation XDF.

Uses scenario-file timing to label EEG windows as LOW / MODERATE / HIGH,
then runs the calibrated model and reports per-block and overall metrics:
  - Accuracy (argmax) for binary (LOW vs HIGH) and 3-class (all blocks)
  - ROC-AUC: binary (P(HIGH) vs LOW/HIGH windows) and macro-OvR 3-class

Usage
-----
    python scripts/evaluate_model_on_adaptation.py \
        --xdf    "C:/data/adaptive_matb/physiology/sub-PSELF/ses-S001/physio/sub-PSELF_ses-S001_task-matb_acq-adaptation_physio_old1.xdf" \
        --scenario "experiment/scenarios/adaptive_automation_pself_c1_8min.txt" \
        --model-dir "C:/data/adaptive_matb/models/PSELF" \
        --offset 12.0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from build_mwl_training_dataset import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)
from eeg import EegPreprocessor, extract_windows, slice_block  # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402

_DEFAULT_REGION_CFG = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"
_ANALYSIS_SRATE = 128.0


def _rolling_z_auc(
    p_high: np.ndarray, is_high: np.ndarray, window: int
) -> tuple[float, float, np.ndarray]:
    """Apply rolling z-normalisation (backward-looking) and return (AUC, Acc@z>0, z_scores).

    Windows before burn-in (first `window` samples) are excluded from metrics.
    """
    z = np.full(len(p_high), np.nan)
    for i in range(window, len(p_high)):
        buf = p_high[i - window : i]
        mu = buf.mean()
        sigma = buf.std()
        z[i] = (p_high[i] - mu) / sigma if sigma > 1e-9 else 0.0
    valid = ~np.isnan(z)
    if valid.sum() < 10:
        return float("nan"), float("nan"), z
    auc = float(roc_auc_score(is_high[valid], z[valid]))
    acc = float(((z[valid] > 0) == is_high[valid].astype(bool)).mean())
    return auc, acc, z


def _parse_adaptation_scenario(scenario_path: Path) -> list[tuple[float, float, str]]:
    """Parse a scenario .txt → (start_s, end_s, level) per block.

    Handles both adaptive_automation and calibration_condition marker paths.
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate personalised model on adaptation XDF.")
    parser.add_argument("--xdf", type=Path, required=True, help="Adaptation .xdf file.")
    parser.add_argument("--scenario", type=Path, required=True,
                        help="Scenario .txt with block markers (path relative to repo root or absolute).")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Dir with pipeline.pkl, selector.pkl, norm_stats.json.")
    parser.add_argument("--offset", type=float, default=12.0,
                        help="Seconds from XDF start to scenario t=0 (default: 12.0).")
    parser.add_argument("--no-norm", action="store_true",
                        help="Skip resting-baseline normalisation (feed raw features directly).")
    parser.add_argument("--rolling-sim", action="store_true",
                        help="After evaluation, simulate rolling z-baseline adaptation strategy.")
    args = parser.parse_args()

    scenario_path = args.scenario if args.scenario.is_absolute() else _REPO_ROOT / args.scenario

    # --- Load model artefacts ---
    pipeline = joblib.load(args.model_dir / "pipeline.pkl")
    selector = joblib.load(args.model_dir / "selector.pkl")
    norm_stats = json.loads((args.model_dir / "norm_stats.json").read_text())
    norm_mean = np.array(norm_stats["mean"], dtype=np.float64)
    norm_std = np.array(norm_stats["std"], dtype=np.float64)
    # Auto-detect n_classes: prefer saved value, fall back to inspecting the clf.
    clf_classes = pipeline.named_steps["clf"].classes_
    n_classes = int(norm_stats.get("n_classes", len(clf_classes)))
    p_high_col = n_classes - 1   # column index for P(HIGH): 1 (binary) or 2 (3-class)
    print(f"Model: {n_classes}-class  (P(HIGH) = column {p_high_col})")

    # --- Load XDF ---
    import pyxdf
    print(f"Loading {args.xdf.name} ...")
    streams, _ = pyxdf.load_xdf(str(args.xdf))

    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found.")

    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        sys.exit(f"ERROR: Channel count {n_ch} != expected {len(expected_channels)}")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    # Decimate to analysis srate if the amplifier ran at a higher rate (e.g. 500 → 128 Hz).
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"  Decimated {actual_srate:.0f}->{_ANALYSIS_SRATE:.0f} Hz (factor={factor})")

    duration_s = eeg_ts[-1] - eeg_ts[0]
    print(f"  EEG: {n_ch} ch, {len(eeg_ts)} samples, {duration_s:.1f}s")

    # --- Preprocess ---
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    # --- Parse scenario block timing ---
    scenario_blocks = _parse_adaptation_scenario(scenario_path)
    if not scenario_blocks:
        sys.exit(f"ERROR: No blocks parsed from {scenario_path}")
    matb_t0 = eeg_ts[0] + args.offset
    print(f"\nScenario offset: {args.offset}s  (matb_t0={matb_t0:.2f})")
    print(f"Block order: {' '.join(lvl[0] for _, _, lvl in scenario_blocks)}\n")

    # --- Region map for feature extraction ---
    region_map = _build_region_map(_DEFAULT_REGION_CFG, expected_channels)

    # --- Evaluate per block ---
    _EVAL_LABEL = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    all_true_3: list[int] = []       # true 3-class label (0/1/2) for every window
    all_p_high: list[float] = []     # P(HIGH) for every window

    print(f"\n{'Block':>6}  {'Level':>8}  {'Windows':>7}  {'P(H) mean':>10}  {'P(H) std':>9}")
    print("-" * 52)

    for i, (start_s, end_s, level) in enumerate(scenario_blocks, 1):
        start_ts = matb_t0 + start_s
        end_ts = matb_t0 + end_s
        start_idx = int(np.searchsorted(eeg_ts, start_ts))
        end_idx = int(np.searchsorted(eeg_ts, end_ts))

        block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)

        if epochs.shape[0] == 0:
            print(f"{i:>6}  {level:>8}  {'0':>7}  {'—':>10}  {'—':>9}")
            continue

        X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
        if args.no_norm:
            X_sel = selector.transform(X)
        else:
            X_norm = (X - norm_mean) / norm_std
            X_sel = selector.transform(X_norm)
        probs = pipeline.predict_proba(X_sel)        # (N, n_classes)
        p_high = probs[:, p_high_col]                # P(HIGH) per window

        true_label = _EVAL_LABEL[level]
        all_true_3.extend([true_label] * len(p_high))
        all_p_high.extend(p_high.tolist())

        print(f"{i:>6}  {level:>8}  {len(epochs):>7}  "
              f"{p_high.mean():>10.3f}  {p_high.std():>9.3f}")

    # --- Summary metrics ---
    if not all_true_3:
        print("\nNo blocks found — cannot compute metrics.")
        return

    y_true = np.array(all_true_3)
    p_high_arr = np.array(all_p_high)
    is_high = (y_true == 2).astype(int)   # 1=HIGH, 0=NOT-HIGH

    print("-" * 52)

    # --- P(HIGH) by level ---
    print("\nP(HIGH) by level:")
    for label, name in [(0, "LOW"), (1, "MODERATE"), (2, "HIGH")]:
        mask = y_true == label
        if mask.any():
            vals = p_high_arr[mask]
            print(f"  {name:<10}  n={mask.sum():>4}  "
                  f"mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"median={float(np.median(vals)):.3f}")

    # --- HIGH vs NOT-HIGH discrimination (all windows) ---
    n_high = int(is_high.sum())
    n_not_high = int((~is_high.astype(bool)).sum())
    auc = float(roc_auc_score(is_high, p_high_arr))
    # Accuracy at threshold 0.5
    acc = float(((p_high_arr >= 0.5) == is_high.astype(bool)).mean())
    chance = max(n_high, n_not_high) / len(y_true)
    print(f"\nHIGH vs NOT-HIGH  n={len(y_true)}  "
          f"(HIGH={n_high}, NOT-HIGH={n_not_high})")
    print(f"  AUC={auc:.3f}   Acc@0.5={acc:.1%}   chance={chance:.1%}")

    # --- Rolling z-baseline simulation ---
    if args.rolling_sim:
        _WINDOWS = [10, 20, 30, 60, 103]
        print("\n--- Rolling z-baseline simulation (backward-looking, no look-ahead) ---")
        print(f"  Fixed threshold baseline: AUC={auc:.3f}  Acc@0.5={acc:.1%}")
        print(f"\n  {'Window':>7}  {'Burn-in':>8}  {'Evaluated':>10}  {'AUC':>7}  {'Acc@z>0':>9}")
        print("  " + "-" * 49)

        best_auc, best_w = -1.0, _WINDOWS[0]
        best_z: np.ndarray = np.full(len(p_high_arr), np.nan)
        for w in _WINDOWS:
            w_auc, w_acc, z = _rolling_z_auc(p_high_arr, is_high, w)
            n_eval = int((~np.isnan(z)).sum())
            pct = n_eval / len(p_high_arr)
            print(f"  {w:>7}  {w:>8}  {n_eval:>7} ({pct:.0%})  {w_auc:>7.3f}  {w_acc:>8.1%}")
            if w_auc > best_auc:
                best_auc, best_w, best_z = w_auc, w, z

        # Per-level z-score stats for best window
        print(f"\n  Best window: {best_w}  (AUC={best_auc:.3f})")
        print(f"  z-score by level (window={best_w}):")
        for label, name in [(0, "LOW"), (1, "MODERATE"), (2, "HIGH")]:
            mask = (y_true == label) & ~np.isnan(best_z)
            if mask.any():
                vals = best_z[mask]
                print(f"    {name:<10}  n={mask.sum():>4}  "
                      f"mean={vals.mean():+.3f}  std={vals.std():.3f}  "
                      f"median={float(np.median(vals)):+.3f}")


if __name__ == "__main__":
    main()
