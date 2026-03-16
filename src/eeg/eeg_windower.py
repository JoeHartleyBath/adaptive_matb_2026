"""Offline epoch extractor for EEG data.

Operates on an already-preprocessed array of shape (n_channels, n_total_samples)
and returns a strided array of epochs (n_epochs, n_channels, n_samples).

This is the offline equivalent of EegInlet.get_window(), using the same
parameter conventions (durations in seconds, srate in Hz).

Usage
-----
    cfg = WindowConfig(window_s=2.0, step_s=0.25, srate=128.0)
    epochs = extract_windows(preprocessed_data, cfg)
    # epochs.shape == (n_epochs, n_channels, 256)
"""

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Constants matching the study design
# ---------------------------------------------------------------------------
WINDOW_S: float = 2.0    # EEGNet temporal receptive field (2 s at 128 Hz → T=256)
STEP_S: float = 0.25     # Overlap step (4 Hz update rate, matches ADR-0003)
SRATE: float = 128.0
WARMUP_S: float = 30.0   # Seconds to discard at block start (filter settle + habituation)


@dataclass
class WindowConfig:
    """Parameters controlling offline epoch extraction.

    Defaults match the study design and must remain consistent between
    build_mwl_training_dataset.py and calibrate_participant.py.
    """
    window_s: float = WINDOW_S
    step_s: float = STEP_S
    srate: float = SRATE

    @property
    def window_samples(self) -> int:
        return int(self.window_s * self.srate)

    @property
    def step_samples(self) -> int:
        return int(self.step_s * self.srate)

    @property
    def warmup_samples(self) -> int:
        return int(WARMUP_S * self.srate)


def extract_windows(
    data: np.ndarray,
    config: WindowConfig | None = None,
) -> np.ndarray:
    """Extract strided windows from a preprocessed EEG array.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_total_samples), float32
        Preprocessed EEG data for a single block.  The caller is responsible
        for trimming warmup samples *before* passing data here (use
        ``config.warmup_samples`` or ``slice_block()``).
    config : WindowConfig, optional
        Windowing parameters.  Defaults to study-design values if omitted.

    Returns
    -------
    epochs : np.ndarray, shape (n_epochs, n_channels, window_samples), float32
        Returns an empty array with shape (0, n_channels, window_samples)
        if insufficient data for even one window.
    """
    if config is None:
        config = WindowConfig()

    n_channels, n_total = data.shape
    W = config.window_samples
    S = config.step_samples

    if n_total < W:
        return np.empty((0, n_channels, W), dtype=np.float32)

    n_epochs = (n_total - W) // S + 1

    # Build strided view then copy to own contiguous memory.
    # data strides: axis-0 = channels, axis-1 = samples
    # Output axes: (epoch, channel, sample-within-window)
    epochs = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_epochs, n_channels, W),
        strides=(
            data.strides[1] * S,   # advancing one epoch = S samples along time
            data.strides[0],       # advancing one channel
            data.strides[1],       # advancing one sample within a window
        ),
        writeable=False,
    ).copy()

    return epochs.astype(np.float32, copy=False)


def slice_block(
    data: np.ndarray,
    start_sample: int,
    end_sample: int,
    config: WindowConfig | None = None,
) -> np.ndarray:
    """Slice a block from a continuous recording and discard warmup.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_total_samples)
        Full continuous preprocessed recording.
    start_sample : int
        Sample index of the block START marker.
    end_sample : int
        Sample index of the block END marker (exclusive).
    config : WindowConfig, optional

    Returns
    -------
    block : np.ndarray, shape (n_channels, n_block_samples)
        Block data with warmup removed.
    """
    if config is None:
        config = WindowConfig()
    start_sample = start_sample + config.warmup_samples
    end_sample = min(end_sample, data.shape[1])
    if start_sample >= end_sample:
        return np.empty((data.shape[0], 0), dtype=np.float32)
    return data[:, start_sample:end_sample].astype(np.float32, copy=False)
