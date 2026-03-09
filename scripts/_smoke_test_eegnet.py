"""Quick smoke test for EEG windower and EEGNet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from eeg import WindowConfig, extract_windows, slice_block

cfg = WindowConfig()
data = np.random.randn(128, 5000).astype("float32")
epochs = extract_windows(data, cfg)
print(f"Windower: input (128,5000) -> epochs {epochs.shape}")
# (5000 - 1000) // 125 + 1 = 33
assert epochs.shape == (33, 128, 1000), f"unexpected shape {epochs.shape}"

block = slice_block(data, 0, 5000, cfg)
print(f"slice_block: warmup trim -> {block.shape}")

from ml import EEGNet, HIGH_CLASS, LABEL_MAP, N_CLASSES
import torch

model = EEGNet(n_channels=128, n_times=1000)
n_params = sum(p.numel() for p in model.parameters())
print(f"EEGNet instantiated: {n_params:,} parameters")

x = torch.randn(4, 1, 128, 1000)
logits = model(x)
assert logits.shape == (4, 3), f"bad logit shape {logits.shape}"
print(f"Forward pass: {x.shape} -> logits {logits.shape}")

scores = model.overload_score(x)
assert scores.shape == (4,)
assert 0.0 <= scores.min().item() <= scores.max().item() <= 1.0
print(f"overload_score: {scores.shape}, values in [{scores.min():.3f}, {scores.max():.3f}]")

for mode in ("none", "head_only", "late_layers", "full"):
    model2 = EEGNet(n_channels=128, n_times=1000)
    model2.set_freeze_mode(mode)
    trainable = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"  freeze_mode={mode:<12} trainable_params={trainable:,}")

print(f"\nLABEL_MAP={LABEL_MAP}  HIGH_CLASS={HIGH_CLASS}  N_CLASSES={N_CLASSES}")
print("\nAll checks passed.")
