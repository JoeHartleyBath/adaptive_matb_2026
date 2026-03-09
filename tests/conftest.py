"""conftest.py – session-scoped pytest fixtures for the synthetic pipeline tests.

test_pipeline_synthetic.py was originally a standalone script whose stages shared
state through local variables.  These session-scoped fixtures replicate that shared
state so the individual test functions can receive their inputs via pytest injection.

Fixture dependency chain
------------------------
    rng ──────────────────────────────────────────────────┐
                                                           ▼
    _pipeline_tmp ──► dataset_path  (writes synthetic HDF5)
                  └─► model_dir     (empty directory for model artefacts)
                                        │  (test_pretraining writes weights here)
                                        ▼
                          pretrained_path  (path only; file written by test_pretraining)

NOTE: pytest collects and runs tests within a file in definition order, so
test_pretraining always executes before test_calibration uses pretrained_path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Constants that match the real pipeline (kept in sync with test_pipeline_synthetic.py)
_N_CHANNELS: int = 128
_N_TIMES: int = 1000          # 2 s × 500 Hz
_EPOCHS_PER_CLASS: int = 40
_N_CLASSES: int = 3           # LOW / MODERATE / HIGH


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _write_synthetic_hdf5(
    path: Path,
    participant_ids: list[str],
    rng: np.random.Generator,
) -> None:
    """Write a minimal synthetic dataset.h5 with the schema expected by MwlDataset."""
    channel_names = [f"CH{i}" for i in range(_N_CHANNELS)]

    with h5py.File(path, "w") as f:
        f.attrs["preprocessing_config_hash"] = "synthetic"
        f.attrs["window_s"] = 2.0
        f.attrs["step_s"] = 0.25
        f.attrs["srate"] = 500.0
        f.attrs["n_channels"] = _N_CHANNELS
        f.attrs["channel_names"] = json.dumps(channel_names)
        f.attrs["built_at"] = "synthetic"

        pgrp = f.create_group("participants")
        for pid in participant_ids:
            all_epochs, all_labels = [], []
            for label_idx in range(_N_CLASSES):
                epochs = rng.standard_normal(
                    (_EPOCHS_PER_CLASS, _N_CHANNELS, _N_TIMES)
                ).astype(np.float32)
                labels = np.full(_EPOCHS_PER_CLASS, label_idx, dtype=np.int64)
                all_epochs.append(epochs)
                all_labels.append(labels)
            grp = pgrp.create_group(pid)
            grp.create_dataset(
                "epochs", data=np.concatenate(all_epochs, axis=0), compression="gzip"
            )
            grp.create_dataset("labels", data=np.concatenate(all_labels, axis=0))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Deterministic RNG shared across all pipeline test stages."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def _pipeline_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Single temporary directory reused across all pipeline stages."""
    return tmp_path_factory.mktemp("pipeline")


@pytest.fixture(scope="session")
def dataset_path(_pipeline_tmp: Path, rng: np.random.Generator) -> Path:
    """Path to a synthetic HDF5 dataset written once for the test session."""
    path = _pipeline_tmp / "dataset.h5"
    _write_synthetic_hdf5(path, ["P001", "P002"], rng)
    return path


@pytest.fixture(scope="session")
def model_dir(_pipeline_tmp: Path) -> Path:
    """Empty directory where pretraining writes its artefacts."""
    d = _pipeline_tmp / "models"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="session")
def pretrained_path(model_dir: Path) -> Path:
    """Path where test_pretraining writes eegnet_pretrained.pt.

    The file does not exist when this fixture is created — it is written by
    test_pretraining, which pytest runs (in file order) before test_calibration.
    """
    return model_dir / "eegnet_pretrained.pt"
