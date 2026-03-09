"""MWL training module: EEGNet architecture and dataset utilities.

Public API
----------
    from src.python.training import EEGNet, MwlDataset, LABEL_MAP, N_CLASSES

Labels
------
    LOW = 0, MODERATE = 1, HIGH = 2  (3-class cross-entropy)
    Inference overload score = softmax(logits)[:, HIGH_CLASS]
"""

from .eegnet import EEGNet, HIGH_CLASS
from .dataset import MwlDataset, LABEL_MAP, N_CLASSES

__all__ = [
    "EEGNet",
    "HIGH_CLASS",
    "MwlDataset",
    "LABEL_MAP",
    "N_CLASSES",
]
