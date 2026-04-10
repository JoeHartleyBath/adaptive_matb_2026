"""Online feature extraction wrapper for real-time MWL estimation.

Bridges the gap between the real-time EEG pipeline (single window at a time)
and ``extract_features()`` which expects batch format ``(N, C, T)``.
Handles PSD computation, region-map setup, and IAF band edges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import welch

from .extract_features import (
    FIXED_BANDS,
    _build_region_map,
    extract_features,
    iaf_bands,
)

# Default region-definition config (ANT Neuro 128-ch, shared with VR-TSST).
_DEFAULT_REGION_CFG = Path("C:/vr_tsst_2025/config/eeg_feature_extraction.yaml")


class OnlineFeatureExtractor:
    """Stateless single-window feature extractor for real-time inference.

    Parameters
    ----------
    channel_labels : list[str]
        Ordered channel names matching the LSL inlet (128 for ANT Neuro).
    srate : float
        Sampling rate in Hz (128.0 after down-sampling).
    iaf : float | None
        Individual alpha frequency.  If *None*, uses fixed 10.0 Hz default.
    region_cfg : Path | None
        YAML config defining channel → region mapping.
        Defaults to the shared VR-TSST region config.
    """

    def __init__(
        self,
        channel_labels: list[str],
        srate: float = 128.0,
        iaf: float | None = None,
        region_cfg: Path | None = None,
    ) -> None:
        self.srate = srate

        cfg_path = region_cfg or _DEFAULT_REGION_CFG
        self.region_map = _build_region_map(cfg_path, channel_labels)

        self.bands = FIXED_BANDS

        self._feat_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str] | None:
        """Feature names from the most recent ``compute()`` call (or *None*)."""
        return self._feat_names

    def compute(self, window: np.ndarray) -> np.ndarray:
        """Extract features from a single EEG window.

        Parameters
        ----------
        window : ndarray, shape ``(C, T)``
            Preprocessed (filtered + CAR) EEG window.

        Returns
        -------
        features : ndarray, shape ``(n_features,)``
            Feature vector (~54 elements).
        """
        if window.ndim != 2:
            raise ValueError(
                f"Expected (C, T) array, got shape {window.shape}"
            )

        epochs = window[np.newaxis, :, :]  # (1, C, T)

        freqs, psd = welch(
            epochs,
            fs=self.srate,
            nperseg=min(256, epochs.shape[-1]),
            axis=-1,
        )

        X, names = extract_features(
            epochs, psd, freqs, self.region_map, self.bands, srate=self.srate,
        )
        self._feat_names = names
        return X[0]  # (n_features,)
