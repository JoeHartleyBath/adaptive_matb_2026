"""update_session_baseline.py

Recompute resting-state normalisation statistics from a short baseline
recording taken at the START of the adaptation session, and overwrite
norm_stats.json in the model directory.

The feature selector (selector.pkl) and classifier (pipeline.pkl) are
NOT changed.  Only norm_mean and norm_std shift to reflect the
participant's EEG state at the time of the adaptation session.

Motivation
----------
Self-pilot data (sub-PSELF, 2026-03-27) showed a 3.6× increase in >3σ
feature outliers between calibration and adaptation sessions separated by
~30 min, causing P(HIGH) saturation across all workload levels.  This
script addresses that by re-anchoring the normalisation reference before the
adaptation blocks begin.  See ADR-0005 for full evidence and rationale.

Usage
-----
    python scripts/session/update_session_baseline.py \\
        --xdf   "C:/data/.../ses-S002/physio/sub-P01_ses-S002_..._baseline.xdf" \\
        --model-dir "C:/data/adaptive_matb/models/P01" \\
        --duration 60

The XDF should contain a short eyes-open resting recording (default: use
first 60 s of EEG data in the file).

A backup of the original norm_stats.json is saved as norm_stats_pretrain.json
before any overwrite.  If a backup already exists, it is not overwritten so
the original training baseline is always preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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
_WARN_Z_SHIFT = 2.0   # Warn if new mean feature z vs old baseline shifts by this much


def _compute_baseline_stats(
    xdf_path: Path,
    duration_s: float,
    region_cfg: Path,
    ch_names: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract features from the first *duration_s* of EEG in *xdf_path*.

    Returns (mean, std, n_windows).
    """
    import pyxdf  # local import — not all environments have it on PATH

    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found in XDF.")

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(ch_names):
        sys.exit(
            f"ERROR: XDF has {n_ch} channels, model expects {len(ch_names)}."
        )

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    # Decimate to analysis srate if needed
    if len(eeg_ts) > 1:
        actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if actual_srate > _ANALYSIS_SRATE * 1.1:
            factor = int(round(actual_srate / _ANALYSIS_SRATE))
            eeg_data = eeg_data[:, ::factor]
            eeg_ts = eeg_ts[::factor]
            print(f"  Decimated {actual_srate:.0f}→{_ANALYSIS_SRATE:.0f} Hz (factor={factor})")

    # Clip to requested duration
    n_samples = int(duration_s * _ANALYSIS_SRATE)
    if eeg_data.shape[1] < n_samples:
        print(
            f"  WARNING: XDF only contains {eeg_data.shape[1]} samples "
            f"({eeg_data.shape[1]/_ANALYSIS_SRATE:.1f} s) — using all of it."
        )
    else:
        eeg_data = eeg_data[:, :n_samples]
        eeg_ts = eeg_ts[:n_samples]

    print(f"  EEG: {n_ch} ch, {eeg_data.shape[1]} samples, "
          f"{eeg_data.shape[1]/_ANALYSIS_SRATE:.1f}s used for baseline")

    # Preprocess
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    # Extract features
    region_map = _build_region_map(region_cfg, ch_names)
    start_idx = 0
    end_idx = preprocessed.shape[1]
    block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
    epochs = extract_windows(block, WINDOW_CONFIG)

    if epochs.shape[0] < 5:
        sys.exit(
            f"ERROR: Only {epochs.shape[0]} windows extracted — "
            "recording too short for a reliable baseline."
        )

    X, _ = _extract_feat(epochs, _ANALYSIS_SRATE, region_map)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-12] = 1.0   # guard against zero-variance features

    return mean, std, epochs.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update norm_stats.json from a fresh resting baseline recording."
    )
    parser.add_argument("--xdf", type=Path, required=True,
                        help="XDF file containing the resting baseline EEG.")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Model directory (contains norm_stats.json to update).")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Seconds of EEG from start of file to use (default: 60).")
    args = parser.parse_args()

    norm_path = args.model_dir / "norm_stats.json"
    backup_path = args.model_dir / "norm_stats_pretrain.json"

    if not norm_path.exists():
        sys.exit(f"ERROR: {norm_path} not found. Is --model-dir correct?")

    print(f"Model dir : {args.model_dir}")
    print(f"XDF       : {args.xdf.name}")
    print(f"Duration  : {args.duration} s")
    print()

    # Load existing norm stats (for comparison / sanity check)
    existing = json.loads(norm_path.read_text())
    old_mean = np.array(existing["mean"])
    old_std = np.array(existing["std"])
    n_classes = existing.get("n_classes", None)

    # Backup original if not already done
    if not backup_path.exists():
        backup_path.write_text(norm_path.read_text())
        print(f"  Backed up original → {backup_path.name}")
    else:
        print(f"  Backup already exists ({backup_path.name}) — not overwriting.")

    # Load channel names
    ch_names = _load_eeg_metadata(_REPO_ROOT)

    # Compute new stats from resting baseline
    print(f"\nExtracting features from baseline recording ...")
    new_mean, new_std, n_windows = _compute_baseline_stats(
        args.xdf, args.duration, _DEFAULT_REGION_CFG, ch_names
    )
    print(f"  Windows extracted      : {n_windows}")
    print(f"  New mean range         : {new_mean.min():.4f} to {new_mean.max():.4f}")
    print(f"  New std  range         : {new_std.min():.4f} to {new_std.max():.4f}")

    # --- Sanity check: how far has the distribution shifted? ---
    # Express new mean in terms of the old normalisation
    shift_z = ((new_mean - old_mean) / old_std)
    abs_shift = float(np.abs(shift_z).mean())
    print(f"\nSanity check:")
    print(f"  Mean |z-shift| of feature distribution : {abs_shift:.3f} σ (old baseline)")
    if abs_shift > _WARN_Z_SHIFT:
        print(
            f"  WARNING: Distribution has shifted {abs_shift:.2f}σ from the "
            "training baseline — this is large and consistent with EEG "
            "non-stationarity.  Refreshing is the right action."
        )
    else:
        print(
            f"  Distribution shift is modest ({abs_shift:.2f}σ) — refresh "
            "is precautionary but still recommended."
        )

    # --- Overwrite norm_stats.json ---
    new_stats: dict = {
        "mean": new_mean.tolist(),
        "std": new_std.tolist(),
    }
    if n_classes is not None:
        new_stats["n_classes"] = n_classes
    norm_path.write_text(json.dumps(new_stats))
    print(f"\n  norm_stats.json updated in {args.model_dir}")
    print(
        f"  Start mwl_estimator.py now — it will load "
        "the refreshed baseline automatically."
    )


if __name__ == "__main__":
    main()
