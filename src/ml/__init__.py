"""MWL training module: EEGNet architecture and dataset utilities.

Public API
----------
    from src.python.training import EEGNet, MwlDataset, LABEL_MAP, N_CLASSES

Labels
------
    LOW = 0, MODERATE = 1, HIGH = 2  (3-class cross-entropy)
    Inference overload score = softmax(logits)[:, HIGH_CLASS]
"""

try:
    from .eegnet import EEGNet, HIGH_CLASS
except ImportError:  # torch not installed — EEGNet-dependent scripts fail at use
    EEGNet = None  # type: ignore[assignment,misc]
    HIGH_CLASS = 2

try:
    from .dataset import MwlDataset, LABEL_MAP, N_CLASSES
except ImportError:  # optional heavy deps
    MwlDataset = None  # type: ignore[assignment,misc]
    LABEL_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
    N_CLASSES = 3

from .pretrain_loader import PretrainDataDir

__all__ = [
    "EEGNet",
    "HIGH_CLASS",
    "MwlDataset",
    "LABEL_MAP",
    "N_CLASSES",
    "PretrainDataDir",
]
