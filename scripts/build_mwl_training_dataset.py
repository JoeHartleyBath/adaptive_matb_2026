"""Build the MWL EEGNet training dataset from raw XDF recordings.

Usage
-----
    python scripts/build_mwl_training_dataset.py [--raw-dir PATH] [--out-dir PATH]

If --raw-dir / --out-dir are omitted the script falls back to values in
``config/paths.yaml`` (see ``config/paths.example.yaml``).

Input layout
------------
    {raw_dir}/training/{pid}/
        Any number of .xdf files.  Each file must contain exactly one
        calibration block (one workload level, one continuous 9-minute run).
        The workload level is detected from the LSL marker stream inside
        the file — not from the filename — using the canonical marker:
            STUDY/V0/calibration/{LEVEL}/START

Output
------
    {out_dir}/dataset.h5     HDF5 file (structure documented in dataset.py)
    {out_dir}/build_log.json Run metadata (participant list, epoch counts, etc.)

Nothing written here is committed to git (data_root is outside the repo).

Preprocessing contract
----------------------
Identical to the live inference path:
  - Causal Butterworth bandpass 0.5–40 Hz, order 4
  - Causal IIR notch 50 Hz, Q=30
  - Common Average Reference
All parameters come from config/eeg_metadata.yaml and must not be changed
after the first dataset build without reprocessing from scratch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyxdf
import yaml

# ---------------------------------------------------------------------------
# src/ on sys.path so we can import package modules
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg import EegPreprocessingConfig, EegPreprocessor, WindowConfig, extract_windows, slice_block  # noqa: E402
from ml.dataset import LABEL_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Marker patterns
# ---------------------------------------------------------------------------
# Matches both old STUDY/V0/calibration/LEVEL/EVENT and the actual format
# STUDY/V0/calibration_condition/N/block_NN/LEVEL/EVENT emitted by the scenario files.
_MARKER_RE = re.compile(
    r"STUDY/V0/calibration(?:_condition/\d+/block_\d+)?/(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
)

# ---------------------------------------------------------------------------
# Canonical preprocessing config (single source of truth for all training)
# ---------------------------------------------------------------------------
PREPROCESSING_CONFIG = EegPreprocessingConfig(
    bp_low_hz=0.5,
    bp_high_hz=40.0,
    bp_order=4,
    notch_freq=50.0,
    notch_quality=30.0,
    apply_car=True,
    srate=128.0,
)

WINDOW_CONFIG = WindowConfig(window_s=2.0, step_s=0.25, srate=128.0)


def _config_hash(cfg: EegPreprocessingConfig) -> str:
    """Stable 8-char hex hash of preprocessing parameters for traceability."""
    s = (
        f"bp_low={cfg.bp_low_hz},bp_high={cfg.bp_high_hz},"
        f"bp_order={cfg.bp_order},notch={cfg.notch_freq},"
        f"notch_q={cfg.notch_quality},car={cfg.apply_car},srate={cfg.srate}"
    )
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _resolve_paths(yaml_path: Path) -> dict:
    """Load config/paths.yaml and resolve ${data_root} substitutions."""
    text = yaml_path.read_text()
    cfg = yaml.safe_load(text)
    data_root = cfg.get("data_root", "")
    for key, val in cfg.items():
        if isinstance(val, str):
            cfg[key] = val.replace("${data_root}", data_root)
    return cfg


def _load_eeg_metadata(repo_root: Path) -> list[str]:
    path = repo_root / "config" / "eeg_metadata.yaml"
    with open(path) as f:
        meta = yaml.safe_load(f)
    return meta["channel_names"]


# ---------------------------------------------------------------------------
# XDF helpers
# ---------------------------------------------------------------------------

def _find_stream(streams: list, stream_type: str):
    """Return the first stream matching stream_type, or None."""
    for s in streams:
        if s["info"]["type"][0] == stream_type:
            return s
    return None


def _merge_eeg_streams(streams: list):
    """Return a single EEG stream dict, merging dual-amp setups if needed.

    The ANT eego dual-amp setup streams two 66-channel EEG streams, each
    containing 64 electrode ('ref') channels plus 1 trigger and 1 counter.
    This function extracts only the electrode channels from each stream and
    concatenates them (sorted by stream name for reproducibility) to produce
    a single 128-channel EEG stream dict.  For single-amp recordings the
    original stream is returned unchanged.
    """
    eeg_streams = [s for s in streams if s["info"]["type"][0] == "EEG"]
    if not eeg_streams:
        return None
    if len(eeg_streams) == 1:
        return eeg_streams[0]

    # Sort by stream name so ordering is deterministic across recordings.
    eeg_streams = sorted(eeg_streams, key=lambda s: s["info"]["name"][0])

    parts = []
    for s in eeg_streams:
        ts = np.array(s["time_series"])  # (n_samples, n_ch)
        try:
            ch_desc = s["info"]["desc"][0]["channels"][0]["channel"]
            ref_idx = [i for i, ch in enumerate(ch_desc) if ch["type"][0] == "ref"]
        except (KeyError, IndexError):
            # No channel descriptor — take all channels
            ref_idx = list(range(ts.shape[1]))
        parts.append(ts[:, ref_idx])

    # Align sample counts (streams may differ by ±1 sample at boundaries).
    min_samples = min(p.shape[0] for p in parts)
    merged_ts = np.concatenate([p[:min_samples] for p in parts], axis=1)

    reference = eeg_streams[0]
    merged = {
        "info": dict(reference["info"]),
        "time_series": merged_ts,
        "time_stamps": np.array(reference["time_stamps"])[:min_samples],
    }
    merged["info"] = dict(reference["info"])  # shallow copy
    merged["info"]["channel_count"] = [str(merged_ts.shape[1])]
    return merged


def _parse_markers(marker_stream) -> list[tuple[float, str]]:
    """Return list of (timestamp, marker_string) from a marker stream."""
    if marker_stream is None:
        return []
    result = []
    for ts, sample in zip(
        marker_stream["time_stamps"], marker_stream["time_series"]
    ):
        result.append((float(ts), str(sample[0])))
    return result


def _find_block_bounds(
    markers: list[tuple[float, str]],
    level: str,
) -> tuple[float, float] | None:
    """Return (start_ts, end_ts) for the given block level, or None.

    When multiple blocks share the same level (multi-block calibration
    scenarios) this returns the bounds of the *first* matching block.
    Use :func:`_extract_all_blocks` to retrieve every block.
    """
    start_ts = end_ts = None
    for ts, text in markers:
        m = _MARKER_RE.match(text.split("|")[0])
        if m and m.group("level") == level:
            if m.group("event") == "START" and start_ts is None:
                start_ts = ts
            elif m.group("event") == "END" and start_ts is not None and end_ts is None:
                end_ts = ts
    if start_ts is None or end_ts is None:
        return None
    return start_ts, end_ts


def _detect_level(markers: list[tuple[float, str]]) -> str | None:
    """Detect workload level from the first START marker."""
    for _, text in markers:
        m = _MARKER_RE.match(text.split("|")[0])
        if m and m.group("event") == "START":
            return m.group("level")
    return None


def _extract_all_blocks(
    markers: list[tuple[float, str]],
) -> list[tuple[float, float, str]]:
    """Extract every labelled block from markers.

    Returns a list of ``(start_ts, end_ts, level)`` tuples in chronological
    order, covering all levels and all repeated blocks within each level.
    """
    open_starts: dict[str, float] = {}
    result: list[tuple[float, float, str]] = []
    for ts, text in markers:
        m = _MARKER_RE.match(text.split("|")[0])
        if not m:
            continue
        level = m.group("level")
        if m.group("event") == "START":
            open_starts[level] = ts
        elif m.group("event") == "END" and level in open_starts:
            result.append((open_starts.pop(level), ts, level))
    return result


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_xdf(
    xdf_path: Path,
    expected_channels: list[str],
    prep_config: EegPreprocessingConfig,
    win_config: WindowConfig,
) -> tuple[np.ndarray, int, str] | None:
    """Load, preprocess, and window one XDF file.

    Returns
    -------
    (epochs, label, level_str)  or  None on failure.
    """
    print(f"  Loading {xdf_path.name} ...", end=" ", flush=True)
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"FAILED (load error: {exc})")
        return None

    eeg_stream = _merge_eeg_streams(streams)
    marker_stream = _find_stream(streams, "Markers")

    if eeg_stream is None:
        print("SKIPPED (no EEG stream)")
        return None

    # ---- channel verification ----
    n_ch_file = int(eeg_stream["info"]["channel_count"][0])
    if n_ch_file != len(expected_channels):
        print(
            f"SKIPPED (channel count mismatch: file has {n_ch_file}, "
            f"expected {len(expected_channels)})"
        )
        return None

    # ---- EEG data: (n_samples, n_channels) → (n_channels, n_samples) ----
    eeg_data: np.ndarray = np.array(
        eeg_stream["time_series"], dtype=np.float32
    ).T  # (n_channels, n_samples)
    eeg_ts: np.ndarray = np.array(eeg_stream["time_stamps"])  # (n_samples,)

    # ---- actual srate from timestamps ----
    if len(eeg_ts) < 2:
        print("SKIPPED (too few samples)")
        return None
    actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
    if actual_srate > prep_config.srate * 1.1:
        factor = int(round(actual_srate / prep_config.srate))
        eeg_data = eeg_data[:, ::factor]
        eeg_ts = eeg_ts[::factor]
        print(f"(decimated {actual_srate:.0f}→{prep_config.srate:.0f} Hz, factor={factor}) ", end="", flush=True)
    elif abs(actual_srate - prep_config.srate) > 5.0:
        print(f"SKIPPED (srate mismatch: {actual_srate:.1f} Hz)")
        return None

    # ---- detect block level and bounds ----
    markers = _parse_markers(marker_stream)
    level = _detect_level(markers)
    if level is None:
        print("SKIPPED (no calibration START marker found)")
        return None

    bounds = _find_block_bounds(markers, level)
    if bounds is None:
        print(f"SKIPPED (missing START or END marker for {level})")
        return None
    start_ts, end_ts = bounds

    # ---- convert timestamps → sample indices ----
    start_idx = int(np.searchsorted(eeg_ts, start_ts))
    end_idx = int(np.searchsorted(eeg_ts, end_ts))

    if end_idx - start_idx < win_config.warmup_samples + win_config.window_samples:
        print(f"SKIPPED (block too short after warmup trim)")
        return None

    # ---- preprocessing (causal, same as live path) ----
    preprocessor = EegPreprocessor(prep_config)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    # ---- slice block and discard warmup ----
    block = slice_block(preprocessed, start_idx, end_idx, win_config)

    # ---- windowing ----
    epochs = extract_windows(block, win_config)
    if epochs.shape[0] == 0:
        print("SKIPPED (no complete windows after slicing)")
        return None

    label = LABEL_MAP[level]
    print(f"OK  level={level}  epochs={epochs.shape[0]}")
    return epochs, label, level


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    # ---- resolve paths ----
    paths_yaml = _REPO_ROOT / "config" / "paths.yaml"
    if not paths_yaml.exists():
        sys.exit(
            "config/paths.yaml not found. Copy config/paths.example.yaml and edit it."
        )
    path_cfg = _resolve_paths(paths_yaml)

    raw_dir: Path = args.raw_dir or (Path(path_cfg["raw_dir"]) / "training")
    out_dir: Path = args.out_dir or (Path(path_cfg["processed_dir"]) / "training")

    if not raw_dir.exists():
        sys.exit(f"raw_dir not found: {raw_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_hdf5 = out_dir / "dataset.h5"
    out_log = out_dir / "build_log.json"

    # ---- channel list ----
    expected_channels = _load_eeg_metadata(_REPO_ROOT)
    print(f"Expected channels: {len(expected_channels)} (NA-271 cap)")

    # ---- discover XDF files ----
    xdf_files = sorted(raw_dir.rglob("*.xdf"))
    if not xdf_files:
        sys.exit(f"No .xdf files found under {raw_dir}")
    print(f"Found {len(xdf_files)} XDF files\n")

    # ---- process each file, group by participant ----
    # Participant ID comes from the immediate parent directory name
    participant_data: dict[str, dict[str, list]] = {}  # pid → level → [epochs]
    skipped = 0

    for xdf_path in xdf_files:
        pid = xdf_path.parent.name
        result = process_xdf(
            xdf_path, expected_channels, PREPROCESSING_CONFIG, WINDOW_CONFIG
        )
        if result is None:
            skipped += 1
            continue
        epochs, label, level = result
        participant_data.setdefault(pid, {})
        participant_data[pid].setdefault(level, [])
        participant_data[pid][level].append(epochs)

    if not participant_data:
        sys.exit("No valid files processed. Check raw_dir layout and XDF contents.")

    # ---- write HDF5 ----
    print(f"\nWriting {out_hdf5} ...")
    cfg_hash = _config_hash(PREPROCESSING_CONFIG)
    log: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing_config_hash": cfg_hash,
        "window_s": WINDOW_CONFIG.window_s,
        "step_s": WINDOW_CONFIG.step_s,
        "srate": WINDOW_CONFIG.srate,
        "n_channels": len(expected_channels),
        "warmup_s": WINDOW_CONFIG.warmup_samples / WINDOW_CONFIG.srate,
        "participants": {},
        "files_skipped": skipped,
    }

    with h5py.File(out_hdf5, "w") as f:
        # File-level metadata
        f.attrs["preprocessing_config_hash"] = cfg_hash
        f.attrs["window_s"] = WINDOW_CONFIG.window_s
        f.attrs["step_s"] = WINDOW_CONFIG.step_s
        f.attrs["srate"] = WINDOW_CONFIG.srate
        f.attrs["n_channels"] = len(expected_channels)
        f.attrs["channel_names"] = json.dumps(expected_channels)
        f.attrs["built_at"] = datetime.now(timezone.utc).isoformat()

        pgrp = f.create_group("participants")

        for pid, level_dict in sorted(participant_data.items()):
            all_epochs_list = []
            all_labels_list = []

            for level, epoch_list in level_dict.items():
                combined = np.concatenate(epoch_list, axis=0)
                labels = np.full(combined.shape[0], LABEL_MAP[level], dtype=np.int64)
                all_epochs_list.append(combined)
                all_labels_list.append(labels)

            if not all_epochs_list:
                continue

            epochs_arr = np.concatenate(all_epochs_list, axis=0)
            labels_arr = np.concatenate(all_labels_list, axis=0)

            grp = pgrp.create_group(pid)
            grp.create_dataset("epochs", data=epochs_arr, compression="gzip")
            grp.create_dataset("labels", data=labels_arr)

            counts = {
                lvl: int(np.sum(labels_arr == LABEL_MAP[lvl]))
                for lvl in ["LOW", "MODERATE", "HIGH"]
            }
            log["participants"][pid] = {
                "total_epochs": int(epochs_arr.shape[0]),
                "class_counts": counts,
            }
            print(f"  {pid}: {epochs_arr.shape[0]} epochs  {counts}")

    # ---- write build log ----
    with open(out_log, "w") as f:
        json.dump(log, f, indent=2)

    total_epochs = sum(v["total_epochs"] for v in log["participants"].values())
    print(
        f"\nDone. {len(participant_data)} participants, "
        f"{total_epochs} total epochs → {out_hdf5}"
    )
    print(f"Build log → {out_log}")


if __name__ == "__main__":
    main()
