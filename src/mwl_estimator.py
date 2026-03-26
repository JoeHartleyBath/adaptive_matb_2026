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
# Dual-stream inlet (real-time stream merge for dual-amp setup)
# ---------------------------------------------------------------------------

def _get_ref_channel_indices(info: pylsl.StreamInfo) -> list[int]:
    """Return indices of 'ref'-type channels from an LSL StreamInfo descriptor.

    The ANT eego amplifier labels electrode channels with type='ref' in the
    stream XML.  Falls back to all channels if no typed descriptor is found.
    """
    try:
        ch_elem = info.desc().child("channels").first_child()
        ref_idx: list[int] = []
        i = 0
        while ch_elem.name() == "channel":
            if ch_elem.child_value("type").strip().lower() == "ref":
                ref_idx.append(i)
            ch_elem = ch_elem.next_sibling()
            i += 1
        if ref_idx:
            return ref_idx
    except Exception:
        pass
    return list(range(info.channel_count()))


class _DualEegInlet:
    """Real-time inlet for a dual-amplifier EEG setup.

    Resolves two LSL EEG streams whose names *contain* ``stream_name_substr``,
    pulls from both, extracts 'ref'-type channels from each, and concatenates
    along the channel axis.  Streams are sorted by name (alphabetical) so the
    channel ordering matches the offline ``_merge_eeg_streams()`` used during
    training (amp-A channels precede amp-B channels).

    Exposes ``pull_chunk()`` and ``get_window()`` with the same shapes
    as ``EegInlet`` so the inference loop is unchanged.
    """

    def __init__(
        self,
        stream_name_substr: str,
        stream_srate: float,
        buffer_s: float,
    ) -> None:
        self.substr = stream_name_substr
        self.stream_srate = stream_srate
        self.n_channels = 0
        self._inlets: list[pylsl.StreamInlet] = []
        self._ref_idx: list[list[int]] = []
        self._buf_n = int(buffer_s * stream_srate)
        self._buffer = np.zeros((0, self._buf_n), dtype=np.float32)
        self._timestamps = np.zeros(self._buf_n, dtype=np.float64)
        self._write_ptr = 0
        self.is_connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        """Resolve two EEG streams matching the substring and open inlets."""
        log.info("Dual-inlet: resolving EEG streams containing '%s' ...", self.substr)
        # minimum=2 waits until at least 2 EEG streams are visible on the network,
        # preventing premature return after only one amp is discovered.
        all_streams = pylsl.resolve_stream("type", "EEG", 2, timeout)
        matched = sorted(
            [s for s in all_streams if self.substr in s.name()],
            key=lambda s: s.name(),
        )
        if len(matched) < 2:
            log.error(
                "Dual-inlet: need ≥2 EEG streams matching '%s', found %d. "
                "Are both amplifiers streaming?",
                self.substr, len(matched),
            )
            return False
        if len(matched) > 2:
            log.warning(
                "Found %d EEG streams matching '%s'; using first 2 (sorted by name).",
                len(matched), self.substr,
            )
            matched = matched[:2]

        total_ch = 0
        for s in matched:
            inlet = pylsl.StreamInlet(s)
            info = inlet.info()
            ref_idx = _get_ref_channel_indices(info)
            log.info(
                "  [A/B] %s — %d total ch, using %d ref ch",
                s.name(), info.channel_count(), len(ref_idx),
            )
            self._inlets.append(inlet)
            self._ref_idx.append(ref_idx)
            total_ch += len(ref_idx)

        self.n_channels = total_ch
        self._buffer = np.zeros((self.n_channels, self._buf_n), dtype=np.float32)
        self.is_connected = True
        log.info("Dual inlet ready: %d channels total.", self.n_channels)
        return True

    def pull_chunk(self, max_samples: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Pull from both inlets, merge ref channels, update ring buffer.

        Returns ``(merged, timestamps)``; both empty if either inlet is dry.
        """
        parts: list[np.ndarray | None] = []
        ts_arrs: list[np.ndarray | None] = []

        for inlet, ref_idx in zip(self._inlets, self._ref_idx):
            data, ts = inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
            if not ts:
                parts.append(None)
                ts_arrs.append(None)
            else:
                arr = np.asarray(data, dtype=np.float32).T  # (n_ch, n_samp)
                parts.append(arr[ref_idx, :])
                ts_arrs.append(np.asarray(ts, dtype=np.float64))

        if parts[0] is None or parts[1] is None:
            return np.array([]), np.array([])

        n_samp = min(parts[0].shape[1], parts[1].shape[1])
        if n_samp == 0:
            return np.array([]), np.array([])

        merged = np.concatenate([p[:, :n_samp] for p in parts], axis=0)
        ts_out = ts_arrs[0][:n_samp]

        # Vectorised ring-buffer update
        end = self._write_ptr + n_samp
        if end <= self._buf_n:
            self._buffer[:, self._write_ptr:end] = merged
            self._timestamps[self._write_ptr:end] = ts_out
        else:
            first = self._buf_n - self._write_ptr
            self._buffer[:, self._write_ptr:] = merged[:, :first]
            self._timestamps[self._write_ptr:] = ts_out[:first]
            rem = n_samp - first
            self._buffer[:, :rem] = merged[:, first:]
            self._timestamps[:rem] = ts_out[first:]
        self._write_ptr = (self._write_ptr + n_samp) % self._buf_n

        return merged, ts_out

    def get_window(self, window_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the most recent *window_s* seconds from the ring buffer."""
        n_samp = int(window_s * self.stream_srate)
        idx = np.arange(self._write_ptr - n_samp, self._write_ptr) % self._buf_n
        return self._buffer[:, idx].copy(), self._timestamps[idx].copy()


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

    # ---- EEG inlet (dual-amp merge) + preprocessor ----
    # _DualEegInlet resolves both ANT eego streams by substring match, extracts
    # ref channels from each, and concatenates to 128 ch — mirroring offline
    # _merge_eeg_streams() so the model receives the same channel layout it was
    # trained on.
    inlet = _DualEegInlet(
        stream_name_substr=args.stream_name,
        stream_srate=args.stream_srate,
        buffer_s=max(10.0, _WINDOW_S + 2.0),
    )
    preproc_cfg = EegPreprocessingConfig(srate=_ANALYSIS_SRATE)

    # Decimation factor (stream srate → analysis srate)
    decim_factor = int(round(args.stream_srate / _ANALYSIS_SRATE))
    if decim_factor < 1:
        decim_factor = 1
    need_decimate = decim_factor > 1
    log.info(
        "Stream @ %.0f Hz → analysis @ %.0f Hz  (decimate ×%d)",
        args.stream_srate, _ANALYSIS_SRATE, decim_factor,
    )

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
    if not inlet.connect(timeout=args.timeout):
        log.error("Could not connect to EEG streams — aborting.")
        sys.exit(1)

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

        # ---- signal quality check (on raw decimated data) ----
        # Checked BEFORE filtering: the raw window amplitude reflects electrode
        # contact quality and amplifier health without filter-transient artefacts.
        # The ANT eego streams in V (not µV).  Clean DC-coupled EEG has RMS
        # ~1e-3 – 0.1 V.  Thresholds:  > 1e-4 V (1 alive channel), < 100 V.
        rms = float(np.sqrt(np.mean(window_raw ** 2)))
        quality = 1.0 if 1e-4 < rms < 100.0 else 0.0

        # ---- preprocess (BP → notch → CAR) ----
        # Per-window fresh preprocessor, but pre-warmed from the first sample
        # of each channel so the IIR filter starts in steady-state for that
        # DC level — this eliminates the large onset transient that would
        # otherwise push post-filter RMS far above the quality threshold.
        win_preprocessor = EegPreprocessor(preproc_cfg)
        win_preprocessor.initialize_filters(n_channels, prewarm=window_raw[:, 0])
        window = win_preprocessor.process(window_raw)

        # Temporary diagnostic — remove once RMS values are confirmed
        if n_inferences < 5:
            rms_filtered = float(np.sqrt(np.mean(window ** 2)))
            log.info(
                "[DIAG #%d] raw_rms=%.2f  filtered_rms=%.2f  quality=%.1f",
                n_inferences, rms, rms_filtered, quality,
            )

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
    _repo_root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--eeg-config",
        default=str(_repo_root / "config" / "eeg_metadata.yaml"),
        help="Path to EEG metadata YAML (channel names)",
    )
    p.add_argument(
        "--region-config",
        default=str(_repo_root / "config" / "eeg_feature_extraction.yaml"),
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
