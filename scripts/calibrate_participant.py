"""Fine-tune a pretrained EEGNet for a single participant + ROC threshold.

Usage
-----
    python scripts/calibrate_participant.py \\
        --pretrained /path/to/eegnet_pretrained.pt \\
        --xdf-dir    /raw/calibration/P001/ \\
        --pid        P001 \\
        [--mode      all]       # none | head_only | late_layers | full | all

Runs all four adaptation conditions by default (--mode all) and writes a
comparison CSV to results/{pid}/adaptation_comparison.csv (committed to git;
model weights are saved outside git).

Required inputs
---------------
  --pretrained   Path to the pretrained weights saved by train_mwl_model.py.
  --xdf-dir      Directory containing .xdf files for this one participant.
                 Each file must hold exactly one calibration block (the
                 workload level is detected from LSL markers, same as the
                 dataset builder).
  --pid          Participant identifier string (used for output naming).

Optional
--------
  --mode         Which fine-tuning conditions to run.  Default "all".
  --test-frac    Fraction of each block's windows held out for evaluation
                 (temporal split: last N% of each block).  Default 0.20.
  --epochs       Maximum fine-tuning epochs per condition.  Default 100.
  --patience     Early stopping patience.  Default 15.
  --device       cuda | mps | cpu.  Default: auto-detect.
  --seed         Default 42.

Outputs
-------
  {data_root}/models/{pid}/eegnet_{mode}.pt      Fine-tuned weights
  {data_root}/models/{pid}/{mode}_threshold.json Overload threshold from ROC
  results/{pid}/adaptation_comparison.csv         Metrics table (committed)
"""

from __future__ import annotations

import argparse
import csv
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

import pyxdf  # noqa: E402
import yaml   # noqa: E402

from eeg import (  # noqa: E402
    EegPreprocessingConfig,
    EegPreprocessor,
    WindowConfig,
    extract_windows,
    slice_block,
)
from ml import EEGNet, HIGH_CLASS  # noqa: E402
from ml.dataset import LABEL_MAP, N_CLASSES  # noqa: E402

# Reuse the same path helpers and XDF utilities from the dataset builder
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from build_mwl_training_dataset import (  # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _detect_level,
    _find_block_bounds,
    _find_stream,
    _load_eeg_metadata,
    _parse_markers,
    _resolve_paths,
)

# Learning rates per freeze mode (conservative; fine-tuning only)
_LR_MAP: dict[str, float] = {
    "none": 0.0,        # no training
    "head_only": 1e-4,
    "late_layers": 5e-5,
    "full": 1e-5,
}

ALL_MODES = ("none", "head_only", "late_layers", "full")


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


def _load_xdf_block(
    xdf_path: Path,
    expected_channels: list[str],
) -> tuple[np.ndarray, str] | None:
    """Load and preprocess one XDF block.  Returns (epochs, level) or None."""
    print(f"  {xdf_path.name} ...", end=" ", flush=True)
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None

    eeg_stream = _find_stream(streams, "EEG")
    marker_stream = _find_stream(streams, "Markers")

    if eeg_stream is None:
        print("SKIPPED (no EEG stream)")
        return None

    n_ch = int(eeg_stream["info"]["channel_count"][0])
    if n_ch != len(expected_channels):
        print(f"SKIPPED (channel count {n_ch} ≠ {len(expected_channels)})")
        return None

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    markers = _parse_markers(marker_stream)
    level = _detect_level(markers)
    if level is None:
        print("SKIPPED (no START marker)")
        return None

    bounds = _find_block_bounds(markers, level)
    if bounds is None:
        print(f"SKIPPED (missing bounds for {level})")
        return None
    start_ts, end_ts = bounds
    start_idx = int(np.searchsorted(eeg_ts, start_ts))
    end_idx = int(np.searchsorted(eeg_ts, end_ts))

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(eeg_data.shape[0])
    preprocessed = preprocessor.process(eeg_data)

    block = slice_block(preprocessed, start_idx, end_idx, WINDOW_CONFIG)
    epochs = extract_windows(block, WINDOW_CONFIG)
    if epochs.shape[0] == 0:
        print("SKIPPED (no windows)")
        return None

    print(f"OK  level={level}  epochs={epochs.shape[0]}")
    return epochs, level


def _temporal_split(
    epochs: np.ndarray,
    labels: np.ndarray,
    test_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by temporal position: first (1-test_frac) → train, rest → test."""
    n = len(labels)
    split = int(n * (1.0 - test_frac))
    return epochs[:split], labels[:split], epochs[split:], labels[split:]


def _build_loaders(
    train_epochs: np.ndarray,
    train_labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    x = torch.from_numpy(train_epochs).unsqueeze(1)   # (N, 1, C, T)
    y = torch.from_numpy(train_labels)
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


@torch.no_grad()
def _evaluate(
    model: "EEGNet",
    epochs: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict:
    """Evaluate model on a numpy epoch array.  Returns metrics dict."""
    model.eval()
    x = torch.from_numpy(epochs).unsqueeze(1).to(device)   # (N, 1, C, T)
    proba = model.predict_proba(x).cpu().numpy()            # (N, 3)
    preds = proba.argmax(axis=1)

    # Per-class accuracy
    per_class_acc = {}
    for name, idx in LABEL_MAP.items():
        mask = labels == idx
        if mask.sum() > 0:
            per_class_acc[name] = float((preds[mask] == idx).mean())
        else:
            per_class_acc[name] = float("nan")

    # Balanced accuracy
    valid = [v for v in per_class_acc.values() if not np.isnan(v)]
    bal_acc = float(np.mean(valid)) if valid else float("nan")

    # AUC (P(HIGH) vs true HIGH label) — binary
    high_scores = proba[:, HIGH_CLASS]
    binary_truth = (labels == LABEL_MAP["HIGH"]).astype(int)
    auc = _roc_auc(binary_truth, high_scores)

    # Youden's J threshold
    threshold, youden_j = _youden_threshold(binary_truth, high_scores)

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "auc_high_vs_rest": round(auc, 4),
        "threshold_youden": round(threshold, 4),
        "youden_j": round(youden_j, 4),
        "acc_LOW": round(per_class_acc.get("LOW", float("nan")), 4),
        "acc_MODERATE": round(per_class_acc.get("MODERATE", float("nan")), 4),
        "acc_HIGH": round(per_class_acc.get("HIGH", float("nan")), 4),
        "n_test": int(len(labels)),
    }


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC without sklearn using the trapezoid rule on sorted scores."""
    desc_score_indices = np.argsort(scores)[::-1]
    sorted_truth = y_true[desc_score_indices]
    tp = np.cumsum(sorted_truth)
    fp = np.cumsum(1 - sorted_truth)
    tpr = tp / max(tp[-1], 1)
    fpr = fp / max(fp[-1], 1)
    # Add (0, 0) origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    # np.trapezoid in NumPy ≥2.0; np.trapz in earlier versions
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is None:
        raise RuntimeError("numpy trapezoid/trapz not found")
    return float(_trapz(tpr, fpr))


def _youden_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Return (threshold, youden_J) at the optimal Youden's J operating point."""
    thresholds = np.unique(scores)[::-1]
    best_j = -1.0
    best_thresh = 0.5
    n_pos = max(y_true.sum(), 1)
    n_neg = max((1 - y_true).sum(), 1)
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tpr = ((preds == 1) & (y_true == 1)).sum() / n_pos
        fpr = ((preds == 1) & (y_true == 0)).sum() / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_thresh = float(t)
    return best_thresh, float(best_j)


def _finetune(
    model: "EEGNet",
    mode: str,
    train_loader: DataLoader,
    val_tensor: tuple[np.ndarray, np.ndarray],
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> None:
    """In-place fine-tuning; modifies model weights."""
    lr = _LR_MAP[mode]
    model.set_freeze_mode(mode)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or lr == 0.0:
        return  # "none" mode — no training

    optimizer = torch.optim.Adam(trainable, lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_x = torch.from_numpy(val_tensor[0]).unsqueeze(1).to(device)
    val_y = torch.from_numpy(val_tensor[1]).to(device)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--xdf-dir", type=Path, required=True)
    parser.add_argument("--pid", type=str, required=True)
    parser.add_argument(
        "--mode", default="all",
        choices=list(ALL_MODES) + ["all"],
    )
    parser.add_argument("--test-frac", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(args.device) if args.device else _auto_device()
    print(f"Participant: {args.pid}  Device: {device}")

    # ---- paths ----
    paths_yaml = _REPO_ROOT / "config" / "paths.yaml"
    path_cfg: dict = {}
    if paths_yaml.exists():
        path_cfg = _resolve_paths(paths_yaml)
        model_out_dir = Path(path_cfg["data_root"]) / "models" / args.pid
    else:
        # Fall back to a models/ directory inside the repo (not committed).
        # Acceptable for testing; production runs require paths.yaml.
        model_out_dir = _REPO_ROOT / "models" / args.pid
    model_out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = _REPO_ROOT / "results" / args.pid
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- channel list ----
    expected_channels = _load_eeg_metadata(_REPO_ROOT)

    # ---- load calibration XDF files ----
    print(f"\nLoading XDF files from {args.xdf_dir} ...")
    xdf_files = sorted(args.xdf_dir.glob("*.xdf"))
    if not xdf_files:
        sys.exit(f"No .xdf files found in {args.xdf_dir}")

    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for path in xdf_files:
        result = _load_xdf_block(path, expected_channels)
        if result is None:
            continue
        epochs, level = result
        labels = np.full(epochs.shape[0], LABEL_MAP[level], dtype=np.int64)
        all_epochs.append(epochs)
        all_labels.append(labels)

    if not all_epochs:
        sys.exit("No valid blocks loaded.")

    epochs_all = np.concatenate(all_epochs, axis=0)
    labels_all = np.concatenate(all_labels, axis=0)

    # ---- temporal train/test split (per-class to preserve balance) ----
    train_epochs_list, train_labels_list = [], []
    test_epochs_list, test_labels_list = [], []

    for lvl_name, lvl_idx in LABEL_MAP.items():
        mask = labels_all == lvl_idx
        if mask.sum() == 0:
            print(f"Warning: no epochs for class {lvl_name}")
            continue
        e, la, te, tl = _temporal_split(
            epochs_all[mask], labels_all[mask], args.test_frac
        )
        train_epochs_list.append(e)
        train_labels_list.append(la)
        test_epochs_list.append(te)
        test_labels_list.append(tl)

    train_epochs = np.concatenate(train_epochs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0).astype(np.int64)
    test_epochs = np.concatenate(test_epochs_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0).astype(np.int64)

    print(
        f"\nTrain epochs: {len(train_labels)}  "
        f"Test epochs: {len(test_labels)}"
    )

    # A small within-train split for early stopping during fine-tuning
    split = int(len(train_labels) * 0.85)
    ft_epochs, ft_labels = train_epochs[:split], train_labels[:split]
    es_epochs, es_labels = train_epochs[split:], train_labels[split:]

    ft_loader = _build_loaders(ft_epochs, ft_labels, batch_size=32, device=device)

    # ---- determine which modes to run ----
    modes_to_run = list(ALL_MODES) if args.mode == "all" else [args.mode]

    # ---- read model geometry from pretrained weights metadata ----
    # Infer from the weight shapes rather than re-opening HDF5.
    import h5py
    dataset_path = (
        Path(path_cfg["processed_dir"]) / "training" / "dataset.h5"
        if path_cfg.get("processed_dir")
        else None
    )
    if dataset_path is not None and dataset_path.exists():
        with h5py.File(dataset_path, "r") as f:
            n_channels = int(f.attrs["n_channels"])  # type: ignore[arg-type]
            n_times = int(
                float(f.attrs["window_s"]) * float(f.attrs["srate"])  # type: ignore[arg-type]
            )
    else:
        n_channels = len(expected_channels)
        n_times = int(WINDOW_CONFIG.window_s * WINDOW_CONFIG.srate)

    # ---- run each condition ----
    results: list[dict] = []

    for mode in modes_to_run:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")

        # Fresh copy of pretrained weights for each condition
        model = EEGNet(n_channels=n_channels, n_times=n_times).to(device)
        state = torch.load(args.pretrained, map_location=device, weights_only=True)
        model.load_state_dict(state)

        if mode != "none":
            print(f"Fine-tuning ({mode}, lr={_LR_MAP[mode]}) ...")
            _finetune(
                model, mode, ft_loader,
                (es_epochs, es_labels),
                max_epochs=args.epochs,
                patience=args.patience,
                device=device,
            )

        # Save fine-tuned weights
        weight_path = model_out_dir / f"eegnet_{mode}.pt"
        torch.save(model.state_dict(), weight_path)

        # Evaluate on held-out test set
        metrics = _evaluate(model, test_epochs, test_labels, device)
        metrics["mode"] = mode
        metrics["pid"] = args.pid
        results.append(metrics)

        # Save per-participant ROC threshold
        thresh_path = model_out_dir / f"{mode}_threshold.json"
        with open(thresh_path, "w") as f:
            json.dump(
                {
                    "pid": args.pid,
                    "mode": mode,
                    "threshold_overload": metrics["threshold_youden"],
                    "youden_j": metrics["youden_j"],
                    "auc": metrics["auc_high_vs_rest"],
                    "calibrated_at": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        print(
            f"  bal_acc={metrics['balanced_accuracy']}  "
            f"AUC={metrics['auc_high_vs_rest']}  "
            f"threshold={metrics['threshold_youden']}"
        )
        print(f"  Weights → {weight_path}")
        print(f"  Threshold → {thresh_path}")

    # ---- write comparison CSV (committed to git) ----
    csv_path = results_dir / "adaptation_comparison.csv"
    fieldnames = [
        "pid", "mode",
        "balanced_accuracy", "auc_high_vs_rest",
        "acc_LOW", "acc_MODERATE", "acc_HIGH",
        "threshold_youden", "youden_j", "n_test",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nComparison table → {csv_path}")

    # Print summary
    print("\n--- Summary ---")
    print(f"{'mode':<14} {'bal_acc':>8} {'AUC':>8} {'threshold':>10}")
    for r in results:
        print(
            f"{r['mode']:<14} "
            f"{r['balanced_accuracy']:>8.4f} "
            f"{r['auc_high_vs_rest']:>8.4f} "
            f"{r['threshold_youden']:>10.4f}"
        )


if __name__ == "__main__":
    main()
