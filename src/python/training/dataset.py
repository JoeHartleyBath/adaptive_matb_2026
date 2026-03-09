"""PyTorch Dataset for MWL EEGNet training and fine-tuning.

HDF5 file structure (written by build_mwl_training_dataset.py)
--------------------------------------------------------------
    /participants/{pid}/epochs    shape (n_epochs, n_channels, n_samples) float32
    /participants/{pid}/labels    shape (n_epochs,)                       int64
    /attrs                        metadata (channel_names, srate, window_s, step_s,
                                            preprocessing_config_hash)

Label encoding
--------------
    LOW      = 0
    MODERATE = 1
    HIGH     = 2
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

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
    """Mental workload dataset backed by a single HDF5 file.

    Parameters
    ----------
    hdf5_path : str | os.PathLike
        Path to the dataset HDF5 file produced by
        ``scripts/build_mwl_training_dataset.py``.
    participant_ids : list[str] | None
        If given, only load data from these participants.  Useful for
        participant-level train/val/test splits.
    transform : callable | None
        Optional transform applied to each epoch tensor (e.g. for
        data augmentation).  Receives a (1, C, T) float32 tensor.
    """

    def __init__(
        self,
        hdf5_path: str | os.PathLike,
        participant_ids: Sequence[str] | None = None,
        transform=None,
    ) -> None:
        super().__init__()
        self.hdf5_path = str(hdf5_path)
        self.transform = transform

        # Load all epochs + labels into memory at construction time.
        # Datasets are expected to fit in RAM (a few GB at most).
        all_epochs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with h5py.File(self.hdf5_path, "r") as f:
            participant_group: Any = f["participants"]
            available = list(participant_group.keys())
            pids = (
                [p for p in participant_ids if p in available]
                if participant_ids is not None
                else available
            )
            if not pids:
                raise ValueError(
                    f"No matching participants found in {self.hdf5_path}. "
                    f"Available: {available}"
                )

            self.participant_ids: list[str] = pids
            self.metadata: dict = dict(f.attrs)

            for pid in pids:
                grp: Any = participant_group[pid]
                epochs_raw: np.ndarray = np.asarray(grp["epochs"][:])
                labels_raw: np.ndarray = np.asarray(grp["labels"][:]).astype(np.int64)
                all_epochs.append(epochs_raw)
                all_labels.append(labels_raw)

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
        # Re-open to obtain per-participant counts in order
        counts: list[int] = []
        with h5py.File(self.hdf5_path, "r") as f:
            grp: Any = f["participants"]
            for p in self.participant_ids:
                counts.append(int(grp[p]["labels"].shape[0]))

        pid_index = self.participant_ids.index(pid)
        start = int(np.sum(counts[:pid_index]))
        end = start + counts[pid_index]
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
