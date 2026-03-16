"""Load per-participant continuous HDF5 files from VR-TSST export.

Bridges the new continuous format (one .h5 per participant with continuous
preprocessed EEG + segment boundaries) to the epoch-based format expected
by downstream consumers (personalised_logreg.py, ensemble_loso.py, etc.).

Schema expected per file (see export_matb_pretrain_dataset.py):
    /eeg              (n_samples, 128)  float32  — continuous preprocessed
    /task_onsets       (n_tasks,)        int64    — block start sample
    /task_offsets      (n_tasks,)        int64    — block end sample
    /task_labels       (n_tasks,)        int8     — 0=LOW, 2=HIGH
    /task_block_order  (n_tasks,)        int8     — temporal order 0–3
    /forest_onsets     (n_forests,)      int64
    /forest_offsets    (n_forests,)      int64
    /forest_block_order (n_forests,)     int8
    /fixation_onsets   (n_fix,)          int64
    /fixation_offsets  (n_fix,)          int64
    attrs: srate, pid, n_channels, channels (JSON), ...

Usage
-----
    from src.ml.pretrain_loader import PretrainDataDir

    data = PretrainDataDir("C:/vr_tsst_2025/output/matb_pretrain/continuous")
    epochs, labels, block_idx = data.load_task_epochs("P01")
    forest_epochs, forest_blk  = data.load_forest_epochs("P01")
    fix_epochs                 = data.load_fixation_epochs("P01")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from eeg.eeg_windower import WindowConfig, extract_windows, slice_block


# ---------------------------------------------------------------------------
# Defaults matching study design
# ---------------------------------------------------------------------------
_DEFAULT_WIN = WindowConfig(window_s=2.0, step_s=0.5, srate=128.0)

# Binary label map (VR-TSST has no MODERATE)
_LABEL_HIGH = 2
_LABEL_LOW = 0


class PretrainDataDir:
    """Convenience wrapper over a directory of per-participant .h5 files."""

    def __init__(self, root: str | Path, win: WindowConfig | None = None):
        self.root = Path(root)
        self.win = win or _DEFAULT_WIN
        if not self.root.is_dir():
            raise FileNotFoundError(f"Continuous data dir not found: {self.root}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def available_pids(self) -> list[str]:
        """Return sorted list of PIDs that have .h5 files."""
        return sorted(p.stem for p in self.root.glob("P*.h5"))

    def pid_path(self, pid: str) -> Path:
        return self.root / f"{pid}.h5"

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def read_attrs(self, pid: str) -> dict:
        """Return file-level attrs as a plain dict."""
        with h5py.File(self.pid_path(pid), "r") as f:
            return dict(f.attrs)

    def channel_names(self, pid: str | None = None) -> list[str]:
        """Read channel names from a participant file (any will do)."""
        if pid is None:
            pids = self.available_pids()
            if not pids:
                raise FileNotFoundError("No participant files found")
            pid = pids[0]
        with h5py.File(self.pid_path(pid), "r") as f:
            return json.loads(f.attrs["channels"])

    def srate(self, pid: str | None = None) -> float:
        if pid is None:
            pid = self.available_pids()[0]
        with h5py.File(self.pid_path(pid), "r") as f:
            return float(f.attrs["srate"])

    # ------------------------------------------------------------------
    # Epoch loading — task blocks
    # ------------------------------------------------------------------

    def load_task_epochs(
        self,
        pid: str,
        win: WindowConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Window task blocks and return (epochs, labels, block_idx).

        Returns
        -------
        epochs : (N, n_channels, window_samples) float32
        labels : (N,) int64  — 0=LOW, 2=HIGH
        block_idx : (N,) int8  — temporal block order 0–3
        """
        w = win or self.win
        with h5py.File(self.pid_path(pid), "r") as f:
            eeg = f["eeg"][:]                       # (n_samples, n_ch)
            onsets = f["task_onsets"][:]
            offsets = f["task_offsets"][:]
            labels_per_block = f["task_labels"][:]
            block_order = f["task_block_order"][:]

        # eeg_windower expects (n_ch, n_samples)
        eeg_ct = eeg.T.astype(np.float32)

        all_epochs = []
        all_labels = []
        all_block_idx = []

        for i in range(len(onsets)):
            block = slice_block(eeg_ct, int(onsets[i]), int(offsets[i]), w)
            epochs = extract_windows(block, w)
            if epochs.shape[0] == 0:
                continue
            all_epochs.append(epochs)
            all_labels.append(
                np.full(epochs.shape[0], int(labels_per_block[i]), dtype=np.int64))
            all_block_idx.append(
                np.full(epochs.shape[0], int(block_order[i]), dtype=np.int8))

        if not all_epochs:
            n_ch = eeg_ct.shape[0]
            W = w.window_samples
            return (np.empty((0, n_ch, W), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int8))

        return (np.concatenate(all_epochs),
                np.concatenate(all_labels),
                np.concatenate(all_block_idx))

    # ------------------------------------------------------------------
    # Task condition metadata
    # ------------------------------------------------------------------

    def load_task_conditions(self, pid: str) -> list[str]:
        """Return per-block task condition names (same order as onsets).

        Example names: 'HighStress_HighCog1022_Task', 'LowStress_LowCog_Task'.
        """
        with h5py.File(self.pid_path(pid), "r") as f:
            return json.loads(f.attrs["task_conditions"])

    # ------------------------------------------------------------------
    # Epoch loading — forest baselines
    # ------------------------------------------------------------------

    def load_forest_epochs(
        self,
        pid: str,
        win: WindowConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Window Forest1–4 blocks.

        Returns
        -------
        epochs : (M, n_channels, window_samples) float32
        block_idx : (M,) int8  — which forest (0–3)
        """
        w = win or self.win
        with h5py.File(self.pid_path(pid), "r") as f:
            eeg = f["eeg"][:]
            onsets = f["forest_onsets"][:]
            offsets = f["forest_offsets"][:]
            block_order = f["forest_block_order"][:]

        eeg_ct = eeg.T.astype(np.float32)
        all_epochs = []
        all_block_idx = []

        for i in range(len(onsets)):
            block = slice_block(eeg_ct, int(onsets[i]), int(offsets[i]), w)
            epochs = extract_windows(block, w)
            if epochs.shape[0] == 0:
                continue
            all_epochs.append(epochs)
            all_block_idx.append(
                np.full(epochs.shape[0], int(block_order[i]), dtype=np.int8))

        if not all_epochs:
            n_ch = eeg_ct.shape[0]
            W = w.window_samples
            return (np.empty((0, n_ch, W), dtype=np.float32),
                    np.empty((0,), dtype=np.int8))

        return np.concatenate(all_epochs), np.concatenate(all_block_idx)

    # ------------------------------------------------------------------
    # Epoch loading — fixation baselines
    # ------------------------------------------------------------------

    def load_fixation_epochs(
        self,
        pid: str,
        win: WindowConfig | None = None,
    ) -> np.ndarray:
        """Window fixation cross blocks.

        Returns
        -------
        epochs : (K, n_channels, window_samples) float32
        """
        w = win or self.win
        with h5py.File(self.pid_path(pid), "r") as f:
            eeg = f["eeg"][:]
            onsets = f["fixation_onsets"][:]
            offsets = f["fixation_offsets"][:]

        eeg_ct = eeg.T.astype(np.float32)
        all_epochs = []

        for i in range(len(onsets)):
            block = slice_block(eeg_ct, int(onsets[i]), int(offsets[i]), w)
            epochs = extract_windows(block, w)
            if epochs.shape[0] > 0:
                all_epochs.append(epochs)

        if not all_epochs:
            n_ch = eeg_ct.shape[0]
            return np.empty((0, n_ch, w.window_samples), dtype=np.float32)

        return np.concatenate(all_epochs)

    # ------------------------------------------------------------------
    # Bulk loading (for LOSO / cache builders)
    # ------------------------------------------------------------------

    def load_all_task_epochs(
        self,
        pids: Sequence[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Load task epochs for multiple participants.

        Returns
        -------
        epochs_by : {pid: (N, C, T)}
        labels_by : {pid: (N,)}
        """
        if pids is None:
            pids = self.available_pids()

        epochs_by: dict[str, np.ndarray] = {}
        labels_by: dict[str, np.ndarray] = {}
        for pid in pids:
            if not self.pid_path(pid).exists():
                continue
            ep, lab, _ = self.load_task_epochs(pid)
            epochs_by[pid] = ep
            labels_by[pid] = lab
        return epochs_by, labels_by


# ---------------------------------------------------------------------------
# Causal normalisation helpers
# ---------------------------------------------------------------------------

def calibration_norm_features(
    task_X: np.ndarray,
    fix_X: np.ndarray,
    forest_X: np.ndarray,
    forest_bidx: np.ndarray,
) -> np.ndarray:
    """Z-score task features using fixation + Forest0 baseline (causal).

    This is the 'calibration' strategy identified as the best
    online-compatible normalisation in causal_norm_comparison.py (Step 10).
    Uses only data available before the first task block (~300 s).

    Parameters
    ----------
    task_X : (N_task, F) feature matrix to normalise
    fix_X : (N_fix, F) fixation epoch features
    forest_X : (N_forest, F) forest epoch features
    forest_bidx : (N_forest,) block index 0–3

    Returns
    -------
    (N_task, F) calibration-normalised features
    """
    parts: list[np.ndarray] = []
    if fix_X.shape[0] > 0:
        parts.append(fix_X)
    fm = forest_bidx == 0
    if fm.any():
        parts.append(forest_X[fm])
    if not parts:
        return task_X.copy()
    bl = np.concatenate(parts)
    mean = bl.mean(axis=0)
    std = bl.std(axis=0)
    std[std < 1e-12] = 1.0
    return (task_X - mean) / std


def load_baseline_from_cache(
    cache_path: Path,
    pids: list[str],
) -> dict[str, dict[str, np.ndarray]] | None:
    """Load baseline features (fix, forest) from norm comparison cache.

    Returns dict[pid] -> {fix_X, forest_X, forest_bidx}, or None if
    the cache is missing or incomplete.
    """
    if not cache_path.exists():
        return None
    try:
        npz = np.load(cache_path, allow_pickle=False)
        baseline: dict[str, dict[str, np.ndarray]] = {}
        for pid in pids:
            baseline[pid] = {
                "fix_X": npz[f"{pid}_fix_X"],
                "forest_X": npz[f"{pid}_forest_X"],
                "forest_bidx": npz[f"{pid}_forest_bidx"],
            }
        return baseline
    except (KeyError, Exception):
        return None


def prepare_mixed_norm(
    X_by_raw: dict[str, np.ndarray],
    baseline_by: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build two feature dicts for realistic LOSO evaluation.

    Training participants get non-causal pp z-score (full session available
    during offline model building).  The held-out participant gets causal
    calibration normalisation (only early-session baseline available online).

    Parameters
    ----------
    X_by_raw : {pid: (N, F)} raw (un-normalised) task features
    baseline_by : {pid: {fix_X, forest_X, forest_bidx}}

    Returns
    -------
    X_by_train : {pid: (N, F)} pp z-scored features (for training)
    X_by_test  : {pid: (N, F)} calibration-normalised (for held-out test)
    """
    from sklearn.preprocessing import StandardScaler

    X_by_train: dict[str, np.ndarray] = {}
    X_by_test: dict[str, np.ndarray] = {}

    for pid, X_raw in X_by_raw.items():
        # Training version: non-causal pp z-score
        sc = StandardScaler()
        X_by_train[pid] = sc.fit_transform(X_raw)

        # Test version: causal calibration norm
        bl = baseline_by.get(pid)
        if bl is not None:
            X_by_test[pid] = calibration_norm_features(
                X_raw, bl["fix_X"], bl["forest_X"], bl["forest_bidx"]
            )
        else:
            # Fallback: pp z-score (no baseline available)
            X_by_test[pid] = X_by_train[pid].copy()

    return X_by_train, X_by_test
