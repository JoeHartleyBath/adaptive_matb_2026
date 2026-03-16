"""Pretrain EEGNet across training participants.

Usage
-----
    python scripts/train_mwl_model.py [options]

    --dataset PATH     Path to continuous data directory.  Defaults to
                       {processed_dir}/training/continuous from paths.yaml.
    --out-dir PATH     Where to save the model and training log.
                       Defaults to {data_root}/models/.
    --hold-out P1 P2   Participant IDs to exclude from training (test set).
                       These participants are never seen during pretraining.
    --epochs N         Maximum training epochs (default: 200).
    --batch-size N     (default: 64)
    --lr FLOAT         Learning rate (default: 1e-3).
    --patience N       Early stopping patience in epochs (default: 20).
    --device STR       "cuda", "mps", or "cpu" (default: auto-detect).
    --seed N           Random seed (default: 42).

Outputs (all outside git)
--------------------------
    {out_dir}/eegnet_pretrained.pt         Best model weights
    {out_dir}/eegnet_pretrained_log.json   Training metadata and loss curves
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import yaml  # noqa: E402

from ml import EEGNet, MwlDataset  # noqa: E402
from ml.pretrain_loader import PretrainDataDir  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_paths(yaml_path: Path) -> dict:
    text = yaml_path.read_text()
    cfg = yaml.safe_load(text)
    data_root = cfg.get("data_root", "")
    for key, val in cfg.items():
        if isinstance(val, str):
            cfg[key] = val.replace("${data_root}", data_root)
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / n_samples


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    n_classes: int = 3,
) -> tuple[float, float]:
    """Return (mean_loss, auc_high_vs_rest)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_proba: list[np.ndarray] = []
    all_labels: list[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        all_proba.append(proba)
        all_labels.extend(y.cpu().tolist())

    proba_arr = np.concatenate(all_proba, axis=0)   # (N, n_classes)
    labels_arr = np.array(all_labels)

    # AUC: P(HIGH) = class index (n_classes-1) after dense remap vs true HIGH label
    high_scores = proba_arr[:, n_classes - 1]
    binary_truth = (labels_arr == (n_classes - 1)).astype(int)
    order = np.argsort(high_scores)[::-1]
    tp = np.cumsum(binary_truth[order])
    fp = np.cumsum(1 - binary_truth[order])
    tpr = np.concatenate([[0.0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.0], fp / max(fp[-1], 1)])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))  # type: ignore[misc]

    return total_loss / n_samples, auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hold-out", nargs="*", default=[], metavar="PID")
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="PID",
        help="Participant IDs to exclude entirely (e.g. QC failures). "
             "Never used for training or validation.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--f1", type=int, default=8, dest="f1",
                        help="EEGNet F1 (number of temporal filters). Default 8.")
    args = parser.parse_args()

    _set_seed(args.seed)

    # ---- paths ----
    paths_yaml = _REPO_ROOT / "config" / "paths.yaml"
    path_cfg: dict = {}
    if args.dataset is None or args.out_dir is None:
        if not paths_yaml.exists():
            sys.exit(
                "config/paths.yaml not found. "
                "Either create it or supply --dataset and --out-dir explicitly."
            )
        path_cfg = _resolve_paths(paths_yaml)

    dataset_path: Path = args.dataset or (
        Path(path_cfg["processed_dir"]) / "training" / "continuous"
    )
    out_dir: Path = args.out_dir or Path(path_cfg["data_root"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else _auto_device()
    print(f"Device: {device}")

    # ---- read metadata via PretrainDataDir ----
    data_dir = PretrainDataDir(dataset_path)
    n_channels = len(data_dir.channel_names())
    srate = data_dir.srate()
    window_s = data_dir.win.window_s
    n_classes = 2  # VR-TSST binary: LOW / HIGH
    n_times = int(window_s * srate)
    print(f"Dataset: {n_channels} ch × {n_times} samples  n_classes={n_classes}")

    # ---- dataset ----
    all_pids: list[str] = data_dir.available_pids()
    exclude_set = set(args.exclude)
    hold_out_set = set(args.hold_out)
    if exclude_set:
        print(f"Excluding (QC failures): {sorted(exclude_set)}")
        all_pids = [p for p in all_pids if p not in exclude_set]
    train_pids = [p for p in all_pids if p not in hold_out_set]
    val_frac = 0.15  # 15% of training participants → validation

    if not train_pids:
        sys.exit("No training participants after applying --hold-out.")

    # Seeded random val split — avoids systematic alphabetical bias
    # (late PIDs alphabetically are not necessarily representative of the
    # full performance distribution).
    n_val_pids = max(1, int(len(train_pids) * val_frac))
    rng_split = np.random.default_rng(args.seed)
    val_indices = sorted(rng_split.choice(len(train_pids), size=n_val_pids, replace=False))
    val_pids = [train_pids[i] for i in val_indices]
    train_pids_final = [p for p in train_pids if p not in set(val_pids)]

    print(
        f"Participants — train: {len(train_pids_final)}  "
        f"val: {len(val_pids)}  "
        f"hold-out: {len(hold_out_set)}"
    )

    train_dataset = MwlDataset(dataset_path, participant_ids=train_pids_final)
    val_dataset = MwlDataset(dataset_path, participant_ids=val_pids)

    # Remap sparse labels to dense [0, n_classes-1].
    # Needed when the dataset encodes only a subset of the canonical 3-class
    # label space (e.g. VR-TSST export: LOW=0, HIGH=2 → remap to 0, 1).
    unique_labels = sorted(set(train_dataset._labels.tolist()))
    if unique_labels != list(range(n_classes)):
        lmap = {old: new for new, old in enumerate(unique_labels)}
        for ds in (train_dataset, val_dataset):
            ds._labels = np.array([lmap[lb] for lb in ds._labels], dtype=np.int64)
        print(f"Label remap applied: {lmap}")

    # Dataset is balanced (equal LOW / HIGH windows per participant), so
    # unweighted CrossEntropyLoss is appropriate.  Class weighting on a
    # balanced dataset adds noise rather than correcting imbalance.
    criterion = nn.CrossEntropyLoss()

    # Compute per-channel normalisation statistics from training epochs only.
    # shape: (N, C, T) → mean/std over N and T, one value per channel.
    # Preserves sustained amplitude differences across workload conditions
    # (e.g. alpha suppression during HIGH load) while removing
    # between-participant baseline offsets.
    ch_mean = train_dataset._epochs.mean(axis=(0, 2)).astype(np.float32)  # (C,)
    ch_std  = train_dataset._epochs.std(axis=(0, 2)).astype(np.float32)   # (C,)
    ch_std  = np.where(ch_std > 1e-10, ch_std, 1.0)                       # flat-channel guard
    train_dataset._channel_mean = ch_mean
    train_dataset._channel_std  = ch_std
    val_dataset._channel_mean   = ch_mean
    val_dataset._channel_std    = ch_std
    print(
        f"Per-channel train stats computed (C={len(ch_mean)}): "
        f"mean|abs|={np.abs(ch_mean).mean():.3e}, mean std={ch_std.mean():.3e}"
    )

    # Save stats alongside the model so they are available at fine-tune
    # and inference time without re-loading the full training dataset.
    channel_stats_path = out_dir / "channel_stats.npz"
    np.savez(channel_stats_path, mean=ch_mean, std=ch_std)
    print(f"Channel stats: {channel_stats_path}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes, F1=args.f1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience // 2, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"EEGNet parameters: {total_params:,}")

    # ---- training loop ----
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []

    best_weights_path = out_dir / "eegnet_pretrained.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = _eval_epoch(model, val_loader, criterion, device, n_classes=n_classes)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_auc": round(val_auc, 4),
        })

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_auc={val_auc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    print(f"\nBest epoch: {best_epoch}  best_val_loss: {best_val_loss:.4f}")
    print(f"Saved: {best_weights_path}")

    # ---- save log ----
    log = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "train_participants": train_pids_final,
        "val_participants": val_pids,
        "hold_out_participants": list(hold_out_set),
        "excluded_participants": sorted(exclude_set),
        "n_channels": n_channels,
        "n_times": n_times,
        "n_classes": n_classes,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_split": "seeded_random",
        "channel_stats_path": str(channel_stats_path),
        "history": history,
    }
    log_path = out_dir / "eegnet_pretrained_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
