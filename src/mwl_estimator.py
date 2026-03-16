"""Real-time MWL estimator process.

Reads EEG from an LSL inlet, computes features, runs LogReg inference,
and publishes P(overload) to an LSL outlet at ~4 Hz.

Architecture reference: ADR-0003 — Process A (MWL Estimator).

Usage
-----
    python -m mwl_estimator --model-dir models/P001 --stream-name eego

Model directory must contain:
    pipeline.pkl   — joblib: sklearn Pipeline (StandardScaler + LogReg)
    selector.pkl   — joblib: SelectKBest (k=30)
    norm_stats.json — {"mean": [...], "std": [...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pylsl
import yaml

from eeg.eeg_inlet import EegInlet
from eeg.eeg_preprocessing_config import EegPreprocessingConfig
from eeg.eeg_preprocessor import EegPreprocessor
from eeg.eeg_stream_config import EegStreamConfig
from eeg.online_features import OnlineFeatureExtractor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_S = 2.0       # Seconds per feature window
_STEP_S = 0.25        # Update interval (~4 Hz)
_DEFAULT_SRATE = 500.0  # ANT Neuro hardware rate
_ANALYSIS_SRATE = 128.0
_OUTLET_NAME = "MWL"
_OUTLET_TYPE = "MWL"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_artefacts(model_dir: Path) -> dict:
    """Load pipeline, selector, and norm stats from *model_dir*.

    Returns dict with keys: ``pipeline``, ``selector``, ``norm_mean``,
    ``norm_std``.
    """
    pipeline_path = model_dir / "pipeline.pkl"
    selector_path = model_dir / "selector.pkl"
    norm_path = model_dir / "norm_stats.json"

    for p in (pipeline_path, selector_path, norm_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing model artefact: {p}")

    pipeline = joblib.load(pipeline_path)
    selector = joblib.load(selector_path)

    with open(norm_path) as f:
        ns = json.load(f)
    norm_mean = np.asarray(ns["mean"], dtype=np.float64)
    norm_std = np.asarray(ns["std"], dtype=np.float64)
    norm_std[norm_std < 1e-12] = 1.0  # guard against zero-variance

    log.info("Loaded model from %s  (selector k=%d)", model_dir, selector.k)
    return {
        "pipeline": pipeline,
        "selector": selector,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
    }


# ---------------------------------------------------------------------------
# LSL outlet
# ---------------------------------------------------------------------------

def _create_outlet(name: str, source_id: str) -> pylsl.StreamOutlet:
    """Create a 3-channel float32 LSL outlet [mwl, confidence, quality]."""
    info = pylsl.StreamInfo(
        name=name,
        type=_OUTLET_TYPE,
        channel_count=3,
        nominal_srate=1.0 / _STEP_S,
        channel_format=pylsl.cf_float32,
        source_id=source_id,
    )
    # Add channel labels
    chns = info.desc().append_child("channels")
    for label in ("mwl_value", "confidence", "signal_quality"):
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    return pylsl.StreamOutlet(info)


# ---------------------------------------------------------------------------
# Decimation helper
# ---------------------------------------------------------------------------

def _decimate(data: np.ndarray, factor: int) -> np.ndarray:
    """Simple integer decimation (pick every *factor*-th sample).

    No anti-alias filter — the upstream bandpass (0.5–40 Hz) already
    band-limits to well below the Nyquist of the decimated rate.
    """
    if factor <= 1:
        return data
    return data[:, ::factor]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """Connect, infer, publish — runs until interrupted."""

    # ---- channel labels from config ----
    meta_path = Path(args.eeg_config)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    ch_names: list[str] = meta["channel_names"]
    n_channels = len(ch_names)

    # ---- model artefacts ----
    model = load_model_artefacts(Path(args.model_dir))

    # ---- EEG inlet + preprocessor ----
    stream_cfg = EegStreamConfig(
        stream_name=args.stream_name,
        expected_srate=args.stream_srate,
        expected_channels=ch_names,
        buffer_duration_s=max(10.0, _WINDOW_S + 2.0),
    )
    preproc_cfg = EegPreprocessingConfig(srate=_ANALYSIS_SRATE)
    preprocessor = EegPreprocessor(preproc_cfg)

    # Decimation factor (stream srate → analysis srate)
    decim_factor = int(round(args.stream_srate / _ANALYSIS_SRATE))
    if decim_factor < 1:
        decim_factor = 1
    need_decimate = decim_factor > 1
    log.info(
        "Stream @ %.0f Hz → analysis @ %.0f Hz  (decimate ×%d)",
        args.stream_srate, _ANALYSIS_SRATE, decim_factor,
    )

    inlet = EegInlet(stream_cfg)  # preprocessor applied *after* decimation

    # ---- feature extractor ----
    extractor = OnlineFeatureExtractor(
        ch_names, srate=_ANALYSIS_SRATE, region_cfg=Path(args.region_config),
    )

    # ---- LSL outlet ----
    outlet = _create_outlet(
        name=args.outlet_name,
        source_id=f"mwl_estimator_{args.stream_name}",
    )

    # ---- connect to EEG ----
    log.info("Resolving EEG stream '%s' ...", args.stream_name)
    if not inlet.connect(timeout=args.timeout):
        log.error("Could not connect to EEG stream — aborting.")
        sys.exit(1)

    # Initialise preprocessor for the channel count from the stream
    preprocessor.initialize_filters(n_channels)

    # ---- warmup: fill buffer before first inference ----
    window_samples = int(_WINDOW_S * _ANALYSIS_SRATE)
    samples_collected = 0
    log.info("Warming up (need %d analysis-rate samples) ...", window_samples)

    while samples_collected < window_samples:
        chunk, _ = inlet.pull_chunk()
        if chunk.size == 0:
            time.sleep(0.01)
            continue
        n_new = chunk.shape[1]
        if need_decimate:
            n_new = n_new // decim_factor
        samples_collected += n_new

    log.info("Warmup complete — starting inference loop.")

    # ---- inference loop ----
    norm_mean = model["norm_mean"]
    norm_std = model["norm_std"]
    selector = model["selector"]
    pipeline = model["pipeline"]

    t_last = time.perf_counter()
    n_inferences = 0
    _tick_times: list[float] = []

    while True:
        # Pull any new EEG data into the ring buffer
        raw_chunk, _ = inlet.pull_chunk()

        # Throttle to ~4 Hz
        now = time.perf_counter()
        elapsed = now - t_last
        if elapsed < _STEP_S:
            time.sleep(max(0.0, _STEP_S - elapsed - 0.005))
            continue
        t_last = time.perf_counter()

        _t_tick_start = time.perf_counter()

        # ---- extract window from ring buffer ----
        # get_window returns (C, T) at the *stream* rate
        window_raw, _ = inlet.get_window(_WINDOW_S)

        if window_raw.size == 0:
            outlet.push_sample([0.5, 0.0, 0.0])
            continue

        # ---- decimate to analysis rate ----
        if need_decimate:
            window_raw = _decimate(window_raw, decim_factor)

        # ---- preprocess (BP → notch → CAR) ----
        # Use a stateless call to avoid filter state issues with overlapping
        # windows — create a fresh preprocessor per window.
        win_preprocessor = EegPreprocessor(preproc_cfg)
        win_preprocessor.initialize_filters(n_channels)
        window = win_preprocessor.process(window_raw)

        # ---- signal quality check ----
        rms = float(np.sqrt(np.mean(window ** 2)))
        quality = 1.0 if 0.5 < rms < 200.0 else 0.0

        if quality < 0.5:
            outlet.push_sample([0.5, 0.0, quality])
            n_inferences += 1
            continue

        # ---- features ----
        features = extractor.compute(window)  # (54,)

        # ---- calibration normalisation ----
        features_normed = (features - norm_mean) / norm_std

        # ---- feature selection + inference ----
        features_sel = selector.transform(features_normed[np.newaxis, :])
        p_overload = float(pipeline.predict_proba(features_sel)[0, 1])

        # Confidence: distance from decision boundary (0.5), scaled to [0, 1]
        confidence = min(1.0, abs(p_overload - 0.5) * 2.0)

        outlet.push_sample([p_overload, confidence, quality])
        _tick_times.append(time.perf_counter() - _t_tick_start)
        n_inferences += 1

        if n_inferences % 40 == 0:  # ~every 10 s
            _mean_ms = sum(_tick_times) / len(_tick_times) * 1000
            _max_ms = max(_tick_times) * 1000
            log.info(
                "MWL=%.3f  conf=%.2f  qual=%.1f  (n=%d)  "
                "tick=%.1fms avg / %.1fms max",
                p_overload, confidence, quality, n_inferences,
                _mean_ms, _max_ms,
            )
            _tick_times.clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time MWL estimator (Process A, ADR-0003).",
    )
    p.add_argument(
        "--model-dir", required=True,
        help="Directory containing pipeline.pkl, selector.pkl, norm_stats.json",
    )
    p.add_argument(
        "--stream-name", default="eego",
        help="LSL stream name to connect to (default: eego)",
    )
    p.add_argument(
        "--stream-srate", type=float, default=_DEFAULT_SRATE,
        help=f"Expected stream sample rate in Hz (default: {_DEFAULT_SRATE})",
    )
    p.add_argument(
        "--outlet-name", default=_OUTLET_NAME,
        help=f"Name for the MWL LSL outlet (default: {_OUTLET_NAME})",
    )
    p.add_argument(
        "--eeg-config", default="config/eeg_metadata.yaml",
        help="Path to EEG metadata YAML (channel names)",
    )
    p.add_argument(
        "--region-config",
        default="C:/vr_tsst_2025/config/eeg_feature_extraction.yaml",
        help="Path to region-definition YAML",
    )
    p.add_argument(
        "--timeout", type=float, default=30.0,
        help="Seconds to wait for EEG stream (default: 30)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MWL] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)

    # Graceful shutdown on Ctrl+C
    _stop = False

    def _handle_signal(sig, frame):
        nonlocal _stop
        log.info("Caught signal %s — shutting down.", sig)
        _stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        run(args)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("MWL estimator stopped.")


if __name__ == "__main__":
    main()
