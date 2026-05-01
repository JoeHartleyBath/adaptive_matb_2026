"""Shared XDF loading helpers and canonical preprocessing/windowing config.

These were originally embedded in ``scripts/build_mwl_training_dataset.py``.
They are now the single source of truth used by:
  - scripts/session/calibrate_participant.py
  - scripts/analysis/evaluate_model_on_adaptation.py

Preprocessing contract
----------------------
The constants below define the live-inference pipeline and must stay in sync
with ``mwl_estimator.py``.  Do not change them without reprocessing all data.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from .eeg_preprocessing_config import EegPreprocessingConfig
from .eeg_windower import WindowConfig

# ---------------------------------------------------------------------------
# Canonical preprocessing / windowing config
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

# ---------------------------------------------------------------------------
# Marker pattern — matches both old and current scenario marker formats
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(
    r"STUDY/V0/calibration(?:_condition/\d+/block_\d+)?/(?P<level>LOW|MODERATE|HIGH)/(?P<event>START|END)"
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_eeg_metadata(repo_root: Path) -> list[str]:
    """Return channel_names list from config/eeg_metadata.yaml."""
    path = repo_root / "config" / "eeg_metadata.yaml"
    with open(path) as f:
        meta = yaml.safe_load(f)
    return meta["channel_names"]


# ---------------------------------------------------------------------------
# XDF stream helpers
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

    eeg_streams = sorted(eeg_streams, key=lambda s: s["info"]["name"][0])

    parts = []
    for s in eeg_streams:
        ts = np.array(s["time_series"])
        try:
            ch_desc = s["info"]["desc"][0]["channels"][0]["channel"]
            ref_idx = [i for i, ch in enumerate(ch_desc) if ch["type"][0] == "ref"]
        except (KeyError, IndexError):
            ref_idx = list(range(ts.shape[1]))
        parts.append(ts[:, ref_idx])

    min_samples = min(p.shape[0] for p in parts)
    merged_ts = np.concatenate([p[:min_samples] for p in parts], axis=1)

    reference = eeg_streams[0]
    merged = {
        "info": dict(reference["info"]),
        "time_series": merged_ts,
        "time_stamps": np.array(reference["time_stamps"])[:min_samples],
    }
    merged["info"]["channel_count"] = [str(merged_ts.shape[1])]
    return merged


# ---------------------------------------------------------------------------
# Marker helpers
# ---------------------------------------------------------------------------

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

    Returns bounds of the *first* matching block when multiple blocks share
    the same level.  Use :func:`_extract_all_blocks` to retrieve every block.
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
