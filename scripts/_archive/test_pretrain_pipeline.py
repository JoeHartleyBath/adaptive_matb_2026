"""End-to-end sanity test: pretrain on HDF5 subset, fine-tune on held-out participant.

Validates the full VR-TSST → MATB pretraining pipeline without requiring live
XDF calibration files.  Not the production path; see calibrate_participant.py.

Workflow
--------
    # Step 1 – pretrain on P01 + P03 (hold out P05):
    python scripts/train_mwl_model.py \\
        --dataset  C:/vr_tsst_2025/output/matb_pretrain/continuous \\
        --out-dir  C:/vr_tsst_2025/output/matb_pretrain/model \\
        --hold-out 05

    # Step 2 – test fine-tuning adaptation modes on P05:
    python scripts/test_pretrain_pipeline.py \\
        --pretrained C:/vr_tsst_2025/output/matb_pretrain/model/eegnet_pretrained.pt \\
        --dataset    C:/vr_tsst_2025/output/matb_pretrain/continuous \\
        --test-pid   05

Outputs
-------
    results/test_pretrain/results.json   metrics table (committed to git)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ml import EEGNet  # noqa: E402
from ml.pretrain_loader import PretrainDataDir  # noqa: E402


def _apply_norm(
    epochs: np.ndarray,
    ch_mean: np.ndarray,
    ch_std: np.ndarray,
) -> np.ndarray:
    """Normalise a (N, C, T) array using per-channel training-fold stats.

    Each channel c is normalised as ``(x[:,c,:] - ch_mean[c]) / ch_std[c]``.
    Stats must come from the training participants of the current LOO fold
    (loaded from channel_stats.npz saved by train_mwl_model.py) so the
    test participant's own signal never influences normalisation.
    """
    return ((epochs - ch_mean[None, :, None]) / ch_std[None, :, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _load_participant(data_dir: PretrainDataDir, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs (N,C,T) float32, labels (N,) int64) for one participant."""
    epochs, labels, _ = data_dir.load_task_epochs(pid)
    return epochs.astype(np.float32), labels.astype(np.int64)


def _remap_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Map sparse label values to contiguous [0, n_classes-1].

    VR-TSST export uses LOW=0, HIGH=2 (no MODERATE).  CrossEntropyLoss
    requires contiguous indices, so remap to 0, 1.
    Returns remapped labels and the mapping dict.
    """
    unique = sorted(set(labels.tolist()))
    lmap = {old: new for new, old in enumerate(unique)}
    remapped = np.array([lmap[lb] for lb in labels], dtype=np.int64)
    return remapped, lmap


def _gap_stratified_split(
    epochs: np.ndarray,
    labels: np.ndarray,
    test_frac: float,
    rng: np.random.Generator,
    gap_windows: int = 4,
    orig_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split with a temporal gap buffer to prevent data leakage.

    At step_s=0.5 s with a 2 s window, adjacent windows share 75 % of raw
    samples, so a pure random train/test split leaks boundary epochs into both
    sets.  This function excludes from the training pool any epoch whose
    *original* position is within *gap_windows* of any selected test epoch.

    Parameters
    ----------
    epochs, labels :
        Array to split (may be a sub-array from a previous outer split).
    test_frac :
        Fraction of each class to hold out as test.
    rng :
        Seeded numpy random generator.
    gap_windows :
        Number of positions either side of each test epoch to exclude from
        training.  Default 4 ≈ 2 s at step_s=0.5 s.
    orig_indices :
        Original 0..N-1 positions of each row in *epochs* within the full
        participant array.  Pass the ``train_orig_idx`` returned by the outer
        split when performing the inner early-stop split so that gap exclusion
        operates in the same coordinate space as the outer split.  If None,
        assumed to be ``np.arange(len(epochs))``.

    Returns
    -------
    train_e, train_l, test_e, test_l, train_orig_idx, test_orig_idx
    """
    if orig_indices is None:
        orig_indices = np.arange(len(epochs))

    train_sub: list[int] = []
    test_sub:  list[int] = []

    for cls in np.unique(labels):
        cls_sub = np.where(labels == cls)[0]          # sub-array positions
        n_test = max(1, int(len(cls_sub) * test_frac))
        chosen_test_sub = rng.choice(cls_sub, size=n_test, replace=False)

        # Build the set of original positions that must be excluded from training
        chosen_orig = set(int(orig_indices[i]) for i in chosen_test_sub)
        buffered_orig: set[int] = set()
        for p in chosen_orig:
            for delta in range(-gap_windows, gap_windows + 1):
                buffered_orig.add(p + delta)

        # Train: not chosen for test AND not inside any buffer zone
        remaining_sub = np.setdiff1d(cls_sub, chosen_test_sub)
        train_sub.extend(
            int(i) for i in remaining_sub if int(orig_indices[i]) not in buffered_orig
        )
        test_sub.extend(int(i) for i in chosen_test_sub)

    train_arr = np.array(train_sub, dtype=np.intp)
    test_arr  = np.array(test_sub,  dtype=np.intp)

    return (
        epochs[train_arr], labels[train_arr].astype(np.int64),
        epochs[test_arr],  labels[test_arr].astype(np.int64),
        orig_indices[train_arr], orig_indices[test_arr],
    )


def _finetune(
    model: EEGNet,
    mode: str,
    train_epochs: np.ndarray,
    train_labels: np.ndarray,
    val_epochs: np.ndarray,
    val_labels: np.ndarray,
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    ch_mean: np.ndarray,
    ch_std: np.ndarray,
) -> None:
    """Fine-tune model in-place according to freeze mode."""
    model.set_freeze_mode(mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or lr == 0.0:
        return  # "none" — zero-shot eval, no training

    x_tr = torch.from_numpy(_apply_norm(train_epochs, ch_mean, ch_std)).unsqueeze(1)
    y_tr = torch.from_numpy(train_labels)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    x_val = torch.from_numpy(_apply_norm(val_epochs, ch_mean, ch_std)).unsqueeze(1).to(device)
    y_val = torch.from_numpy(val_labels).to(device)

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state: dict | None = None
    patience_ctr = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), y_val).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def _evaluate(
    model: EEGNet,
    epochs: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    ch_mean: np.ndarray,
    ch_std: np.ndarray,
) -> dict:
    """Return balanced accuracy and AUC."""
    model.eval()
    x = torch.from_numpy(_apply_norm(epochs, ch_mean, ch_std)).unsqueeze(1).to(device)
    proba = torch.softmax(model(x), dim=1).cpu().numpy()
    preds = proba.argmax(axis=1)

    # Balanced accuracy
    per_class = []
    for cls in np.unique(labels):
        mask = labels == cls
        per_class.append(float((preds[mask] == cls).mean()))
    bal_acc = float(np.mean(per_class))

    # AUC — P(HIGH) class (index 1 after remap) vs rest
    scores = proba[:, 1]
    truth = (labels == 1).astype(int)
    order = np.argsort(scores)[::-1]
    tp = np.cumsum(truth[order])
    fp = np.cumsum(1 - truth[order])
    tpr = np.concatenate([[0.], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0.], fp / max(fp[-1], 1)])
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(_trapz(tpr, fpr))  # type: ignore[misc]

    return {"bal_acc": round(bal_acc, 4), "auc": round(auc, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_LR_MAP: dict[str, float] = {
    "none":        0.0,
    "head_only":   1e-4,
    "late_layers": 5e-5,
    "full":        1e-5,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="Path to eegnet_pretrained.pt from train_mwl_model.py")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to continuous data directory (per-participant HDF5 files)")
    parser.add_argument("--test-pid", type=str, required=True,
                        help="Participant ID to fine-tune/evaluate (e.g. '05')")
    parser.add_argument("--test-frac", type=float, default=0.10,
                        help="Fraction of each participant's epochs held out for test. "
                             "Keep low when gap_windows>0 to avoid buffer over-consumption "
                             "(default: 0.10; rule of thumb: test_frac*(2*gap+1) < 0.8).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--channel-stats", type=Path, required=True, metavar="NPZ",
        help="Path to channel_stats.npz saved by train_mwl_model.py for this fold. "
             "Contains 'mean' and 'std' arrays (C,) computed from training participants only.",
    )
    parser.add_argument(
        "--gap-windows", type=int, default=3, metavar="N",
        help="Exclude ±N positions around each test epoch from the training pool to "
             "prevent data leakage from overlapping windows (default: 3, which guarantees "
             "zero raw-sample overlap at step_s=0.5s, window_s=2s). Set 0 to disable.",
    )
    parser.add_argument("--f1", type=int, default=8, dest="f1",
                        help="EEGNet F1 (number of temporal filters). Must match the "
                             "pretrained model. Default 8.")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device) if args.device else _auto_device()
    rng = np.random.default_rng(args.seed)
    print(f"Device: {device}")

    # Load per-channel normalisation stats from the training fold.
    # These MUST come from the same fold's train_mwl_model.py run so that
    # the test participant's data is normalised identically to the training data.
    if not args.channel_stats.exists():
        sys.exit(f"[ERROR] --channel-stats file not found: {args.channel_stats}")
    stats = np.load(args.channel_stats)
    ch_mean: np.ndarray = stats["mean"].astype(np.float32)
    ch_std:  np.ndarray = stats["std"].astype(np.float32)
    print(f"Channel stats loaded: {args.channel_stats.name}  (C={len(ch_mean)})")

    # ---- read metadata via PretrainDataDir ----
    data_dir = PretrainDataDir(args.dataset)
    n_channels = len(data_dir.channel_names())
    srate      = data_dir.srate()
    window_s   = data_dir.win.window_s
    n_classes_pretrain = 2  # VR-TSST binary: LOW / HIGH
    available_pids = data_dir.available_pids()
    n_times = int(window_s * srate)

    if args.test_pid not in available_pids:
        sys.exit(
            f"Participant '{args.test_pid}' not found in dataset. "
            f"Available: {available_pids}"
        )

    # ---- load and prepare P{test_pid} ----
    print(f"\nLoading participant {args.test_pid} from {args.dataset.name} ...")
    epochs, labels_raw = _load_participant(data_dir, args.test_pid)
    labels, lmap = _remap_labels(labels_raw)
    n_classes = len(lmap)
    print(f"  {epochs.shape[0]} windows  label remap: {lmap}")

    train_e, train_l, test_e, test_l, train_idx, _ = _gap_stratified_split(
        epochs, labels, args.test_frac, rng, gap_windows=args.gap_windows
    )
    # Inner split: 15% of train → early stopping, gap-buffered in original coordinates
    ft_e, ft_l, es_e, es_l, _, _ = _gap_stratified_split(
        train_e, train_l, 0.15, rng, gap_windows=args.gap_windows, orig_indices=train_idx
    )
    print(
        f"  Train: {len(train_l)}  (fine-tune: {len(ft_l)}, early-stop: {len(es_l)})  "
        f"Test: {len(test_l)}"
    )

    # ---- run each freeze mode ----
    results: list[dict] = []

    for mode in ("none", "head_only", "late_layers", "full"):
        # Fresh pretrained weights each time
        model = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes_pretrain,
            F1=args.f1,
        ).to(device)
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state)

        # If pretrained head dimension differs from fine-tune n_classes, swap it.
        # (n_classes_pretrain == n_classes here since both come from the same HDF5,
        #  but this guard makes the script safe if weights from a 3-class model are
        #  used in future.)
        if n_classes_pretrain != n_classes:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, n_classes).to(device)
            nn.init.xavier_uniform_(model.classifier.weight)
            print(f"  [info] Replaced classifier head: {n_classes_pretrain} → {n_classes} classes")

        if mode != "none":
            _finetune(
                model, mode, ft_e, ft_l, es_e, es_l,
                max_epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=_LR_MAP[mode],
                device=device,
                ch_mean=ch_mean,
                ch_std=ch_std,
            )

        metrics = _evaluate(model, test_e, test_l, device, ch_mean=ch_mean, ch_std=ch_std)
        metrics["mode"] = mode
        results.append(metrics)
        print(f"  {mode:14s}  bal_acc={metrics['bal_acc']:.4f}  auc={metrics['auc']:.4f}")

    # ---- summary table ----
    print("\n" + "=" * 50)
    print(f"{'mode':<14}  {'bal_acc':>8}  {'auc':>8}")
    print("-" * 34)
    for r in results:
        print(f"{r['mode']:<14}  {r['bal_acc']:>8.4f}  {r['auc']:>8.4f}")
    print("=" * 50)
    print("chance level (2-class) = 0.5000")

    # ---- save results ----
    out_dir = _REPO_ROOT / "results" / "test_pretrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "built_at":      datetime.now(timezone.utc).isoformat(),
        "test_pid":      args.test_pid,
        "pretrained":    str(args.pretrained),
        "dataset":       str(args.dataset),
        "seed":          args.seed,
        "test_frac":     args.test_frac,
        "n_train":       int(len(train_l)),
        "n_test":        int(len(test_l)),
        "label_remap":   {str(k): v for k, v in lmap.items()},
        "results":       results,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(log, indent=2))
    print(f"\nResults saved: {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
