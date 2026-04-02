"""Test whether each model is tracking true workload vs just task difficulty.

For each model variant, computes P(HIGH) per 2-s EEG window, then aligns
those windows to MATB tracking performance (center_deviation from the event
log).  Reports:

  r_overall       - Pearson r(P_HIGH, track_dev) across all windows
  r_within_high   - same, restricted to HIGH blocks only
  r_partial       - r(P_HIGH, track_dev) once block-label variance is removed
                    from both signals (residual correlation)

If a model is tracking *actual workload* (not just block difficulty) its
within-block performance correlation (r_within_high, r_partial) should be
positive and clearly larger than the other models.

Usage
-----
    python scripts/correlate_mwl_with_performance.py \
        --xdf  "C:/data/.../adaptation_physio_old1.xdf" \
        --scenario "experiment/scenarios/adaptive_automation_pself_c1_8min.txt" \
        --matb-csv "C:/data/.../22_260326_151644.csv" \
        --model-dirs ws_bin:"C:/data/models/compare/ws_bin" \
                     ws_3cls:"C:/data/models/compare/ws_3cls" \
                     scratch_bin:"C:/data/models/compare/scratch_bin" \
                     scratch_3cls:"C:/data/models/compare/scratch_3cls" \
        --offset 25.17
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

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
from eeg.eeg_windower import WARMUP_S  # noqa: E402

_DEFAULT_REGION_CFG = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"
_ANALYSIS_SRATE = 128.0

_WARMUP_S = WARMUP_S       # 30 s — kept in sync with the rest of the pipeline
_WINDOW_S = WINDOW_CONFIG.window_s   # 2 s
_STEP_S = WINDOW_CONFIG.step_s       # 0.25 s


def _parse_scenario(scenario_path: Path) -> list[tuple[float, float, str]]:
    """Return list of (start_s, end_s, level) tuples for each block."""
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


def _tracking_per_window(
    matb_csv: Path,
    window_midpoints_lsl: np.ndarray,
    half_w: float = _WINDOW_S / 2.0,
) -> np.ndarray:
    """Return mean tracking center_deviation for each EEG window.

    Bins MATB tracking events into [mid - half_w, mid + half_w] intervals by
    direct LSL-time alignment (MATB logtime == EEG LSL clock).

    Returns an array of shape (n_windows,) with NaN where no tracking data
    was available.
    """
    df = pd.read_csv(matb_csv)
    track = df[(df["type"] == "performance") &
               (df["module"] == "track") &
               (df["address"] == "center_deviation")].copy()
    track["value"] = pd.to_numeric(track["value"], errors="coerce")
    track = track.dropna(subset=["value"])

    t_log = track["logtime"].values
    v_log = track["value"].values

    result = np.full(len(window_midpoints_lsl), np.nan)
    for i, mid in enumerate(window_midpoints_lsl):
        mask = (t_log >= mid - half_w) & (t_log < mid + half_w)
        if mask.sum() > 0:
            result[i] = v_log[mask].mean()
    return result


def _partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Pearson r(x, y | z): residuals of x~z and y~z, then correlate.

    Returns (r, p_value).
    """
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 10:
        return float("nan"), float("nan")
    # Regress z out of x and y
    slope_x, inter_x, _, _, _ = stats.linregress(z, x)
    slope_y, inter_y, _, _, _ = stats.linregress(z, y)
    res_x = x - (slope_x * z + inter_x)
    res_y = y - (slope_y * z + inter_y)
    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def _eval_model(
    model_dir: Path,
    preprocessed: np.ndarray,
    eeg_ts: np.ndarray,
    scenario_blocks: list[tuple[float, float, str]],
    matb_t0: float,
    region_map: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one model over all blocks.

    Returns:
        p_high_arr     shape (N,)
        true_label_arr shape (N,)  0=LOW, 1=MOD, 2=HIGH
        window_mid_arr shape (N,)  LSL mid-point timestamps
        block_idx_arr  shape (N,)
    """
    pipeline = joblib.load(model_dir / "pipeline.pkl")
    selector = joblib.load(model_dir / "selector.pkl")
    ns = json.loads((model_dir / "norm_stats.json").read_text())
    norm_mean = np.array(ns["mean"], dtype=np.float64)
    norm_std = np.array(ns["std"], dtype=np.float64)
    clf_classes = pipeline.named_steps["clf"].classes_
    n_classes = int(ns.get("n_classes", len(clf_classes)))
    p_high_col = n_classes - 1

    _EVAL_LABEL = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    all_p_high: list[float] = []
    all_true: list[int] = []
    all_mid: list[float] = []
    all_bidx: list[int] = []

    for bidx, (start_s, end_s, level) in enumerate(scenario_blocks):
        start_ts = matb_t0 + start_s
        end_ts = matb_t0 + end_s
        si = int(np.searchsorted(eeg_ts, start_ts))
        ei = int(np.searchsorted(eeg_ts, end_ts))

        block = slice_block(preprocessed, si, ei, WINDOW_CONFIG)
        epochs = extract_windows(block, WINDOW_CONFIG)
        if epochs.shape[0] == 0:
            continue

        # Compute per-window LSL midpoints:
        # slice_block discards warmup_samples before the first window,
        # so window i starts at: si_actual + i * step_samples
        si_actual = si + WINDOW_CONFIG.warmup_samples
        step_samp = WINDOW_CONFIG.step_samples
        win_samp = WINDOW_CONFIG.window_samples
        for i in range(epochs.shape[0]):
            w_start = si_actual + i * step_samp
            w_end = w_start + win_samp
            w_start_clamped = min(w_start, len(eeg_ts) - 1)
            w_end_clamped = min(w_end, len(eeg_ts) - 1)
            mid_lsl = (eeg_ts[w_start_clamped] + eeg_ts[w_end_clamped]) / 2.0
            all_mid.append(float(mid_lsl))

        X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
        X_norm = (X - norm_mean) / norm_std
        X_sel = selector.transform(X_norm)
        probs = pipeline.predict_proba(X_sel)
        p_high = probs[:, p_high_col]

        all_p_high.extend(p_high.tolist())
        all_true.extend([_EVAL_LABEL[level]] * len(p_high))
        all_bidx.extend([bidx] * len(p_high))

    return (
        np.array(all_p_high),
        np.array(all_true),
        np.array(all_mid),
        np.array(all_bidx),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate per-window P(HIGH) with MATB tracking performance.")
    parser.add_argument("--xdf", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--matb-csv", type=Path, required=True,
                        help="MATB event log CSV (logtime on LSL clock).")
    parser.add_argument("--model-dirs", nargs="+", required=True,
                        metavar="NAME:PATH",
                        help="One or more name:path pairs, e.g. ws_bin:C:/models/ws_bin")
    parser.add_argument("--offset", type=float, default=12.0,
                        help="Seconds from XDF start to scenario t=0.")
    args = parser.parse_args()

    scenario_path = args.scenario if args.scenario.is_absolute() else _REPO_ROOT / args.scenario

    # Parse model dirs
    model_dirs: list[tuple[str, Path]] = []
    for entry in args.model_dirs:
        name, _, path_str = entry.partition(":")
        if not path_str:
            sys.exit(f"ERROR: --model-dirs entries must be name:path, got '{entry}'")
        model_dirs.append((name, Path(path_str)))

    # ── Load & preprocess EEG once ──────────────────────────────────────────
    import pyxdf
    print(f"Loading {args.xdf.name} ...")
    streams, _ = pyxdf.load_xdf(str(args.xdf))

    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found.")

    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        sys.exit(f"ERROR: channel count {n_ch} != expected {len(expected_channels)}")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"  Decimated {actual_srate:.0f}->{_ANALYSIS_SRATE:.0f} Hz")

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)
    print(f"  EEG preprocessed: {preprocessed.shape[1]} samples")

    scenario_blocks = _parse_scenario(scenario_path)
    if not scenario_blocks:
        sys.exit(f"ERROR: No blocks parsed from {scenario_path}")

    matb_t0 = eeg_ts[0] + args.offset
    print(f"  matb_t0={matb_t0:.2f}  blocks: {' '.join(lvl[0] for _, _, lvl in scenario_blocks)}\n")

    # ── Evaluate each model ──────────────────────────────────────────────────
    region_map = _build_region_map(_DEFAULT_REGION_CFG, expected_channels)
    results: dict[str, dict] = {}
    for name, mdir in model_dirs:
        print(f"Running {name} ...")
        p_high, true_label, win_mid, block_idx = _eval_model(
            mdir, preprocessed, eeg_ts, scenario_blocks, matb_t0, region_map)
        results[name] = {
            "p_high": p_high,
            "true_label": true_label,
            "win_mid": win_mid,
            "block_idx": block_idx,
        }

    # All models share the same window midpoints — use first model's
    first_name = list(results.keys())[0]
    win_mid = results[first_name]["win_mid"]
    true_label = results[first_name]["true_label"]

    # ── Compute per-window tracking performance ──────────────────────────────
    print(f"\nBinning tracking performance ({len(win_mid)} windows) ...")
    track_dev = _tracking_per_window(args.matb_csv, win_mid)
    n_valid = int(np.isfinite(track_dev).sum())
    print(f"  Valid windows (tracking data present): {n_valid}/{len(win_mid)}")
    print(f"  Tracking deviation: mean={np.nanmean(track_dev):.2f}  "
          f"std={np.nanstd(track_dev):.2f}  "
          f"max={np.nanmax(track_dev):.2f}\n")

    # ── Correlation table ────────────────────────────────────────────────────
    # Use block label as numeric covariate for partial correlation (0/1/2)
    block_label_num = true_label.astype(float)

    valid = np.isfinite(track_dev)

    print(f"{'Model':<16}  {'r_overall':>10}  {'p':>8}  "
          f"{'r_within_high':>14}  {'p':>8}  "
          f"{'r_partial':>10}  {'p':>8}")
    print("-" * 82)

    for name, d in results.items():
        ph = d["p_high"]

        # Overall r
        mask_ov = valid
        r_ov, p_ov = stats.pearsonr(ph[mask_ov], track_dev[mask_ov])

        # Within-HIGH only
        mask_hi = valid & (true_label == 2)
        if mask_hi.sum() >= 10:
            r_hi, p_hi = stats.pearsonr(ph[mask_hi], track_dev[mask_hi])
        else:
            r_hi, p_hi = float("nan"), float("nan")

        # Partial r (block label regressed out)
        r_pa, p_pa = _partial_r(ph, track_dev, block_label_num)

        print(f"{name:<16}  {r_ov:>+10.3f}  {p_ov:>8.4f}  "
              f"{r_hi:>+14.3f}  {p_hi:>8.4f}  "
              f"{r_pa:>+10.3f}  {p_pa:>8.4f}")

    print()
    print("Interpretation guide:")
    print("  r_overall   : total correlation including between-block difficulty effect")
    print("  r_within_high: within HIGH blocks — does model respond to workload fluctuations?")
    print("  r_partial   : correlation after removing block-label (difficulty) variance")
    print("                Positive = model predicts performance beyond knowing the block label")
    print("                A 'difficulty detector' would show r_partial ~0")


if __name__ == "__main__":
    main()
