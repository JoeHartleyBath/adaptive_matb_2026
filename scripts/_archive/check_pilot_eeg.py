"""Lightweight per-participant EEG signal QC for pilot XDF recordings.

Checks
------
  - Recording duration
  - Per-channel amplitude RMS (µV, raw and preprocessed)
  - Flat / extremely noisy channel detection
  - Optionally: per-feature Z-scores vs calibration norm (--model-dir)

Usage
-----
    python scripts/analysis/check_pilot_eeg.py \\
        --xdf  "C:/data/adaptive_matb/physiology/sub-P001/ses-S001/physio/eeg.xdf" \\
        [--model-dir  "C:/data/adaptive_matb/models/P001"] \\
        [--out  results/figures/P001/S001/eeg_qc.png]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyxdf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg import EegPreprocessor, extract_windows          # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402
from eeg.xdf_loader import (                               # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)

# ---------------------------------------------------------------------------
# Thresholds  (all in µV)
# ---------------------------------------------------------------------------
_FLAT_UV = 0.5       # channels with preprocessed std < this are flagged as flat
_NOISY_UV = 200.0    # channels with preprocessed RMS > this are flagged as noisy
_Z_WARN = 3.0        # feature median |Z| above this is flagged
_DEFAULT_REGION_CFG = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"

# The ANT eego amplifier streams EEG data in volts via LSL.
# Multiply by this factor before displaying or threshold-comparing (volts → µV).
_STREAM_TO_UV = 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight EEG signal QC for pilot XDF recordings."
    )
    parser.add_argument("--xdf", required=True, type=Path, help="Path to .xdf recording.")
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Calibration model directory containing norm_stats.json (enables feature-level QC).",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Save per-channel RMS figure to this path (optional).",
    )
    args = parser.parse_args()

    if not args.xdf.exists():
        sys.exit(f"ERROR: XDF not found: {args.xdf}")

    # ------------------------------------------------------------------
    # Load XDF
    # ------------------------------------------------------------------
    print(f"\nLoading {args.xdf.name} ...", flush=True)
    streams, _ = pyxdf.load_xdf(str(args.xdf))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found in XDF.")

    raw_ts = np.array(eeg_stream["time_series"], dtype=np.float64).T   # (n_channels, n_samples)
    eeg_ts = np.array(eeg_stream["time_stamps"])
    n_ch = raw_ts.shape[0]

    # Compute actual srate from timestamps (more reliable than nominal)
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
    else:
        actual_srate = float(eeg_stream["info"]["nominal_srate"][0])

    target_srate = PREPROCESSING_CONFIG.srate  # 128.0
    decimated = False
    if actual_srate > target_srate * 1.1:
        factor = int(round(actual_srate / target_srate))
        raw_ts = raw_ts[:, ::factor]
        eeg_ts = eeg_ts[::factor]
        decimated = True

    n_samples = raw_ts.shape[1]
    duration_s = n_samples / target_srate

    ch_names = _load_eeg_metadata(_REPO_ROOT)
    if len(ch_names) != n_ch:
        print(
            f"WARNING: metadata has {len(ch_names)} channels but stream has {n_ch}. "
            "Using integer indices."
        )
        ch_names = [str(i) for i in range(n_ch)]

    print(f"  Channels : {n_ch}")
    if decimated:
        print(f"  Srate    : {actual_srate:.0f} Hz → {target_srate:.0f} Hz (decimated x{factor})")
    else:
        print(f"  Srate    : {actual_srate:.1f} Hz")
    print(f"  Duration : {duration_s:.1f} s  ({duration_s / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    print("\nPreprocessing ...", flush=True)
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(n_ch)
    prep_ts = preprocessor.process(raw_ts)   # (n_channels, n_samples)

    # Convert V → µV for display and thresholding
    raw_rms  = np.sqrt(np.mean(raw_ts  ** 2, axis=1)) * _STREAM_TO_UV
    prep_rms = np.sqrt(np.mean(prep_ts ** 2, axis=1)) * _STREAM_TO_UV
    prep_std = prep_ts.std(axis=1) * _STREAM_TO_UV

    # ------------------------------------------------------------------
    # Classify channels
    # ------------------------------------------------------------------
    flat_idx  = [i for i in range(n_ch) if prep_std[i]  < _FLAT_UV]
    noisy_idx = [i for i in range(n_ch) if prep_rms[i] > _NOISY_UV]
    flagged_idx = sorted(set(flat_idx) | set(noisy_idx))

    # Print table: if ≤32 channels show all, else show only flagged
    show_all = n_ch <= 32
    print(f"\n{'Channel':<8}  {'Raw RMS':>10}  {'Prep RMS':>10}  Status")
    print("-" * 46)
    for i, ch in enumerate(ch_names):
        status = ""
        if i in flat_idx:
            status = "FLAT"
        elif i in noisy_idx:
            status = "NOISY"
        if show_all or status:
            print(f"{ch:<8}  {raw_rms[i]:>8.1f} µV  {prep_rms[i]:>8.1f} µV  {status}")

    if not show_all:
        print(f"(showing {len(flagged_idx)} flagged / {n_ch} total channels)")

    flat_names = [ch_names[i] for i in flat_idx]
    noisy_names = [ch_names[i] for i in noisy_idx]
    print(f"\nFlat    ({len(flat_idx)}) : {', '.join(flat_names) or 'none'}")
    print(f"Noisy   ({len(noisy_idx)}) : {', '.join(noisy_names) or 'none'}")
    print(f"Median prep RMS : {np.median(prep_rms):.1f} µV")

    # ------------------------------------------------------------------
    # Feature-level QC (optional)
    # ------------------------------------------------------------------
    if args.model_dir is not None:
        norm_path = args.model_dir / "norm_stats.json"
        if not norm_path.exists():
            print(
                f"\nWARNING: norm_stats.json not found at {norm_path}. "
                "Skipping feature-level QC."
            )
        else:
            print("\nFeature-level QC ...", flush=True)
            ns = json.loads(norm_path.read_text(encoding="utf-8"))
            norm_mean = np.asarray(ns["mean"])
            norm_std = np.asarray(ns["std"])
            norm_std[norm_std < 1e-12] = 1.0

            windows = extract_windows(prep_ts, WINDOW_CONFIG)   # (N, C, T)
            if len(windows) == 0:
                print("WARNING: No windows extracted — recording may be too short.")
            else:
                region_map = _build_region_map(_DEFAULT_REGION_CFG, ch_names)
                X, feat_names = _extract_feat(
                    windows, PREPROCESSING_CONFIG.srate, region_map
                )
                X_z = (X - norm_mean) / norm_std
                z_med = np.median(np.abs(X_z), axis=0)   # per-feature median |Z|
                flagged_feats = [
                    (feat_names[i], float(z_med[i]))
                    for i in range(len(feat_names))
                    if z_med[i] > _Z_WARN
                ]
                print(f"  Windows  : {len(windows)}")
                print(f"  Features : {len(feat_names)}")
                if flagged_feats:
                    flagged_feats.sort(key=lambda x: -x[1])
                    print(
                        f"\n  Features with median |Z| > {_Z_WARN} "
                        "(vs calibration norm — large shift expected for a new participant):"
                    )
                    for fname, z in flagged_feats[:20]:
                        print(f"    {fname:<35}  median|Z| = {z:.2f}")
                    if len(flagged_feats) > 20:
                        print(f"    ... and {len(flagged_feats) - 20} more")
                else:
                    print(f"  All features within {_Z_WARN}σ of calibration norm.")

    # ------------------------------------------------------------------
    # Optional figure
    # ------------------------------------------------------------------
    if args.out is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(prep_rms)
            xs = np.arange(n_ch)
            colours = [
                "red" if prep_rms[order[i]] > _NOISY_UV
                else "orange" if prep_std[order[i]] < _FLAT_UV
                else "steelblue"
                for i in range(n_ch)
            ]

            fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
            axes[0].bar(xs, raw_rms[order], color=colours, alpha=0.8)
            axes[0].set_title("Raw RMS (µV)")
            axes[0].set_xlabel("Channel (sorted by prep RMS)")
            axes[0].set_ylabel("RMS (µV)")

            axes[1].bar(xs, prep_rms[order], color=colours, alpha=0.8)
            axes[1].set_title("Preprocessed RMS (µV)")
            axes[1].set_xlabel("Channel (sorted by prep RMS)")

            fig.suptitle(
                f"EEG QC — {args.xdf.name}\n"
                f"Duration: {duration_s:.0f} s  |  Flat: {len(flat_idx)}  |"
                f"  Noisy: {len(noisy_idx)}  |  Median prep RMS: {np.median(prep_rms):.1f} µV",
                fontsize=9,
            )
            fig.tight_layout()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, dpi=150)
            plt.close(fig)
            print(f"\nFigure saved: {args.out}")
        except ImportError:
            print("WARNING: matplotlib not available — skipping figure.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_bad = len(flagged_idx)
    if n_bad == 0:
        print("\nOVERALL: OK — no flagged channels.")
    else:
        print(f"\nOVERALL: {n_bad} channel(s) flagged — review before calibration.")


if __name__ == "__main__":
    main()
