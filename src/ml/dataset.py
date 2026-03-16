"""PyTorch Dataset for MWL EEGNet training and fine-tuning.

Data source: per-participant continuous HDF5 files in a directory,
read via PretrainDataDir (see ml.pretrain_loader).

Label encoding
--------------
    LOW      = 0
    MODERATE = 1
    HIGH     = 2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .pretrain_loader import PretrainDataDir

# ---------------------------------------------------------------------------
# Label constants – single source of truth inside this module
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, int] = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
}
N_CLASSES: int = 3


class MwlDataset(Dataset):
    """Mental workload dataset backed by per-participant continuous HDF5 files.

    Parameters
    ----------
    data_path : str | os.PathLike
        Path to the continuous data directory containing per-participant
        .h5 files (read via PretrainDataDir).
    participant_ids : list[str] | None
        If given, only load data from these participants.  Useful for
        participant-level train/val/test splits.
    transform : callable | None
        Optional transform applied to each epoch tensor (e.g. for
        data augmentation).  Receives a (1, C, T) float32 tensor.
    """

    def __init__(
        self,
        data_path: str | os.PathLike,
        participant_ids: Sequence[str] | None = None,
        transform=None,
        channel_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        channel_stats : (mean, std) pair of (C,) float32 arrays, optional
            Per-channel normalisation statistics computed from the training
            participants at the start of each LOO fold.  When provided,
            each epoch is normalised as ``(x - mean[c]) / std[c]`` along
            the channel axis, preserving between-condition amplitude
            differences (alpha suppression etc.).  When None, falls back
            to the legacy per-epoch global z-score.
        """
        super().__init__()
        self.data_path = str(data_path)
        self._data_dir = PretrainDataDir(data_path)
        self.transform = transform

        # Per-channel normalisation stats (set here or assigned post-construction)
        if channel_stats is not None:
            self._channel_mean: np.ndarray | None = channel_stats[0].astype(np.float32)
            self._channel_std: np.ndarray | None = channel_stats[1].astype(np.float32)
        else:
            self._channel_mean = None
            self._channel_std = None

        # Load all epochs + labels into memory at construction time.
        # Datasets are expected to fit in RAM (a few GB at most).
        all_epochs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        available = self._data_dir.available_pids()
        pids = (
            [p for p in participant_ids if p in available]
            if participant_ids is not None
            else available
        )
        if not pids:
            raise ValueError(
                f"No matching participants found in {self.data_path}. "
                f"Available: {available}"
            )

        self.participant_ids: list[str] = pids
        self.metadata: dict = {
            "srate": self._data_dir.srate(),
            "n_channels": len(self._data_dir.channel_names()),
            "channels": self._data_dir.channel_names(),
        }
        self._pid_counts: list[int] = []

        for pid in pids:
            epochs_raw, labels_raw, _ = self._data_dir.load_task_epochs(pid)
            all_epochs.append(epochs_raw.astype(np.float32))
            all_labels.append(labels_raw.astype(np.int64))
            self._pid_counts.append(len(labels_raw))

        self._epochs = np.concatenate(all_epochs, axis=0)   # (N, C, T) float32
        self._labels = np.concatenate(all_labels, axis=0)   # (N,) int64

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # EEGNet expects input shape (1, C, T) — add channel-as-colour dim
        epoch = torch.from_numpy(self._epochs[idx]).unsqueeze(0)  # (1, C, T)
        label = torch.tensor(self._labels[idx], dtype=torch.long)

        # Per-channel normalisation using training-fold statistics.
        # channel_mean/std shape: (C,) → broadcast over (1, C, T).
        # Must be set before iterating — assigned by train_mwl_model.py
        # after constructing the dataset for each LOO fold.
        if self._channel_mean is None or self._channel_std is None:
            raise RuntimeError(
                "MwlDataset: channel_stats not set.  Assign _channel_mean and "
                "_channel_std (both (C,) float32 arrays computed from training "
                "participants) before using this dataset in a DataLoader."
            )
        ch_mean = torch.from_numpy(self._channel_mean).view(1, -1, 1)  # (1, C, 1)
        ch_std  = torch.from_numpy(self._channel_std).view(1, -1, 1)   # (1, C, 1)
        epoch = (epoch - ch_mean) / ch_std

        if self.transform is not None:
            epoch = self.transform(epoch)

        return epoch, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def participant_indices(self, pid: str) -> np.ndarray:
        """Return the integer indices in this dataset belonging to *pid*.

        Useful for constructing participant-level samplers or splits.
        """
        pid_index = self.participant_ids.index(pid)
        start = int(np.sum(self._pid_counts[:pid_index]))
        end = start + self._pid_counts[pid_index]
        return np.arange(start, end)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced training sets.

        Returns a (N_CLASSES,) float32 tensor suitable for passing to
        ``nn.CrossEntropyLoss(weight=...)``.
        """
        counts = np.bincount(self._labels, minlength=N_CLASSES).astype(np.float32)
        counts = np.maximum(counts, 1.0)  # avoid division by zero
        weights = 1.0 / counts
        weights = weights / weights.sum() * N_CLASSES  # normalise to mean=1
        return torch.from_numpy(weights)
