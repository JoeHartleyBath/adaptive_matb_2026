"""EEGNet architecture for mental workload classification.

Reference: Lawhern et al. (2018), "EEGNet: A Compact Convolutional Neural
Network for EEG-Based Brain-Computer Interfaces", J Neural Eng 15(5).

Model contract
--------------
Input  : (batch, 1, C, T)  float32
           C = n_channels (128 for NA-271 cap)
           T = window_samples (256 for 2 s at 128 Hz)
Output : (batch, 3)  raw logits  [LOW, MODERATE, HIGH]

Training loss : nn.CrossEntropyLoss (expects raw logits)
Inference     : softmax(logits)[:, HIGH_CLASS]  →  overload score ∈ [0, 1]

Fine-tuning freeze modes
------------------------
  "none"        – use pretrained model as-is (no parameter updates)
  "head_only"   – freeze all layers; unfreeze only self.classifier
  "late_layers" – freeze temporal + depthwise blocks; unfreeze separable + classifier
  "full"        – unfreeze all parameters

Hyperparameters (EEGNet defaults for 128 Hz, 128ch)
---------------------------------------------------
  F1             = 8    temporal filters
  D              = 2    depth multiplier for depthwise conv
  F2             = 16   (= F1 * D) separable filters
  kernel_temporal = 64   (= srate // 2, per paper recommendation)
  dropout_rate   = 0.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Index of the HIGH workload class in the output logits / softmax
HIGH_CLASS: int = 2


class _DepthwiseConv2d(nn.Module):
    """Conv2d with groups=in_channels (depthwise)."""

    def __init__(
        self,
        in_channels: int,
        depth_multiplier: int,
        kernel_size: tuple[int, int],
        padding: int | str = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EEGNet(nn.Module):
    """Compact CNN for 3-class mental workload classification from EEG.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (C).  Default 128 for NA-271 cap.
    n_times : int
        Samples per window (T).  Default 256 (2 s × 128 Hz).
    n_classes : int
        Number of output classes.  Default 3 (LOW / MODERATE / HIGH).
    F1 : int
        Number of temporal filters.  Default 8.
    D : int
        Depth multiplier for the depthwise spatial conv.  Default 2.
    dropout_rate : float
        Dropout probability applied after each pooling step.  Default 0.5.
    srate : float
        Sampling rate in Hz; used to derive temporal kernel size (srate // 2).
    """

    def __init__(
        self,
        n_channels: int = 128,
        n_times: int = 256,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        dropout_rate: float = 0.5,
        srate: float = 128.0,
    ) -> None:
        super().__init__()

        F2 = F1 * D
        kernel_temporal = int(srate // 2)  # 250 for 500 Hz

        # ------------------------------------------------------------------
        # Block 1 – Temporal convolution
        # Conv across time only (spatial kernel height = 1).
        # 'same' padding preserves T.
        # No normalisation here: the input is already per-epoch z-scored
        # (removing cross-subject amplitude differences), so adding
        # InstanceNorm2d over the same (C, T) dims would be redundant and
        # its affine parameters receive near-zero gradients (normalised
        # inputs cancel), flattening the loss surface.
        # ------------------------------------------------------------------
        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                1, F1,
                kernel_size=(1, kernel_temporal),
                padding=(0, kernel_temporal // 2),
                bias=False,
            ),
        )

        # ------------------------------------------------------------------
        # Block 2 – Depthwise spatial convolution
        # Kernel spans full channel dimension (C, 1); no padding.
        # Output spatial dim collapses to 1.
        # ------------------------------------------------------------------
        self.spatial_block = nn.Sequential(
            _DepthwiseConv2d(F1, D, kernel_size=(n_channels, 1), bias=False),
            nn.InstanceNorm2d(F2, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate),
        )

        # ------------------------------------------------------------------
        # Block 3 – Separable convolution
        # Depthwise across time, then pointwise to mix.
        # ------------------------------------------------------------------
        self.separable_block = nn.Sequential(
            # Depthwise temporal
            nn.Conv2d(
                F2, F2,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=F2,
                bias=False,
            ),
            # Pointwise
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.InstanceNorm2d(F2, affine=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate),
        )

        # ------------------------------------------------------------------
        # Classifier head
        # Flatten size computed by a single dry-run forward pass.
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            flat_size = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        x = self.separable_block(x)
        return x.flatten(start_dim=1)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, n_classes)."""
        return self.classifier(self._forward_features(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        return F.softmax(self.forward(x), dim=1)

    def overload_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(HIGH) as a 1-D tensor of shape (batch,).

        This is the continuous scalar fed to the adaptation controller
        and used for per-participant ROC threshold calibration.
        """
        return self.predict_proba(x)[:, HIGH_CLASS]

    # ------------------------------------------------------------------
    # Fine-tuning helpers
    # ------------------------------------------------------------------
    _FREEZE_MODES = ("none", "head_only", "late_layers", "full")

    def set_freeze_mode(self, mode: str) -> None:
        """Configure which parameters are trainable for fine-tuning.

        Parameters
        ----------
        mode : str
            One of: "none", "head_only", "late_layers", "full"
              none        – all parameters frozen (use for zero-shot eval)
              head_only   – only self.classifier unfrozen
              late_layers – self.separable_block + self.classifier unfrozen
              full        – all parameters trainable
        """
        if mode not in self._FREEZE_MODES:
            raise ValueError(
                f"mode must be one of {self._FREEZE_MODES!r}, got {mode!r}"
            )

        # Start fully frozen
        for param in self.parameters():
            param.requires_grad_(False)

        if mode == "none":
            pass  # everything frozen for eval
        elif mode == "head_only":
            for param in self.classifier.parameters():
                param.requires_grad_(True)
        elif mode == "late_layers":
            for param in self.separable_block.parameters():
                param.requires_grad_(True)
            for param in self.classifier.parameters():
                param.requires_grad_(True)
        elif mode == "full":
            for param in self.parameters():
                param.requires_grad_(True)
