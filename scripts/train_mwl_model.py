"""Pretrain EEGNet across training participants.

Usage
-----
    python scripts/train_mwl_model.py [options]

    --dataset PATH     Path to dataset.h5.  Defaults to
                       {processed_dir}/training/dataset.h5 from paths.yaml.
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

from training import EEGNet, MwlDataset  # noqa: E402


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
) -> tuple[float, float]:
    """Return (mean_loss, balanced_accuracy)."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    # Balanced accuracy (mean per-class recall)
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    n_classes = 3
    per_class = []
    for c in range(n_classes):
        mask = labels_arr == c
        if mask.sum() > 0:
            per_class.append((preds_arr[mask] == c).mean())
    bal_acc = float(np.mean(per_class)) if per_class else 0.0

    return total_loss / n_samples, bal_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--hold-out", nargs="*", default=[], metavar="PID")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
        Path(path_cfg["processed_dir"]) / "training" / "dataset.h5"
    )
    out_dir: Path = args.out_dir or Path(path_cfg["data_root"]) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else _auto_device()
    print(f"Device: {device}")

    # ---- dataset ----
    full_dataset = MwlDataset(dataset_path)
    all_pids = full_dataset.participant_ids
    hold_out_set = set(args.hold_out)
    train_pids = [p for p in all_pids if p not in hold_out_set]
    val_frac = 0.15  # 15% of training participants → validation

    if not train_pids:
        sys.exit("No training participants after applying --hold-out.")

    # Per-participant split: last val_frac participants → val
    # Sorted for reproducibility; deterministic given the seed.
    n_val_pids = max(1, int(len(train_pids) * val_frac))
    val_pids = train_pids[-n_val_pids:]
    train_pids_final = train_pids[:-n_val_pids]

    print(
        f"Participants — train: {len(train_pids_final)}  "
        f"val: {len(val_pids)}  "
        f"hold-out: {len(hold_out_set)}"
    )

    train_dataset = MwlDataset(dataset_path, participant_ids=train_pids_final)
    val_dataset = MwlDataset(dataset_path, participant_ids=val_pids)

    class_weights = train_dataset.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # ---- model ----
    # Read n_channels from dataset metadata
    import h5py
    with h5py.File(dataset_path, "r") as f:
        n_channels = int(f.attrs["n_channels"])  # type: ignore[arg-type]
        window_s = float(f.attrs["window_s"])    # type: ignore[arg-type]
        srate = float(f.attrs["srate"])           # type: ignore[arg-type]
    n_times = int(window_s * srate)

    model = EEGNet(n_channels=n_channels, n_times=n_times).to(device)
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
        val_loss, val_bal_acc = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_bal_acc": round(val_bal_acc, 4),
        })

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_bal_acc={val_bal_acc:.3f}"
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
        "n_channels": n_channels,
        "n_times": n_times,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "history": history,
    }
    log_path = out_dir / "eegnet_pretrained_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
