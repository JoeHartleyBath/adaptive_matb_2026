"""Synthetic end-to-end pipeline test.

Tests everything from HDF5 dataset → pretraining → fine-tuning → ROC threshold
using randomly generated data.  XDF loading is the only step not exercised
(it requires the third-party pyxdf library and real files; it is tested
separately when real data are available).

Run from the repo root:
    python scripts/test_pipeline_synthetic.py

Exit code 0 = all stages passed.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch  # noqa: E402

from ml import EEGNet, HIGH_CLASS, LABEL_MAP, N_CLASSES  # noqa: E402
from ml.dataset import MwlDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching the real pipeline
# ---------------------------------------------------------------------------
N_CHANNELS = 128
N_TIMES = 1000       # 2 s × 500 Hz
EPOCHS_PER_CLASS = 40   # small but enough for multiple batches


# ---------------------------------------------------------------------------
# Helper: write a minimal synthetic HDF5 dataset
# ---------------------------------------------------------------------------

def _write_synthetic_hdf5(
    path: Path,
    participant_ids: list[str],
    rng: np.random.Generator,
) -> None:
    """Write a dataset.h5 with the exact schema expected by MwlDataset."""
    import json as _json

    channel_names = [f"CH{i}" for i in range(N_CHANNELS)]

    with h5py.File(path, "w") as f:
        f.attrs["preprocessing_config_hash"] = "synthetic"
        f.attrs["window_s"] = 2.0
        f.attrs["step_s"] = 0.25
        f.attrs["srate"] = 500.0
        f.attrs["n_channels"] = N_CHANNELS
        f.attrs["channel_names"] = _json.dumps(channel_names)
        f.attrs["built_at"] = "synthetic"

        pgrp = f.create_group("participants")
        for pid in participant_ids:
            all_epochs = []
            all_labels = []
            for label_idx in range(N_CLASSES):
                epochs = rng.standard_normal(
                    (EPOCHS_PER_CLASS, N_CHANNELS, N_TIMES)
                ).astype(np.float32)
                labels = np.full(EPOCHS_PER_CLASS, label_idx, dtype=np.int64)
                all_epochs.append(epochs)
                all_labels.append(labels)
            grp = pgrp.create_group(pid)
            grp.create_dataset(
                "epochs", data=np.concatenate(all_epochs, axis=0), compression="gzip"
            )
            grp.create_dataset(
                "labels", data=np.concatenate(all_labels, axis=0)
            )
    print(f"  Wrote synthetic HDF5: {path.name}  "
          f"({len(participant_ids)} participants × {N_CLASSES} classes × "
          f"{EPOCHS_PER_CLASS} epochs)")


# ---------------------------------------------------------------------------
# Stage 1: Dataset round-trip
# ---------------------------------------------------------------------------

def test_dataset(dataset_path: Path) -> None:
    print("\n[Stage 1] MwlDataset round-trip ...")
    ds = MwlDataset(dataset_path)
    assert len(ds) == len(["P001", "P002"]) * N_CLASSES * EPOCHS_PER_CLASS
    x, y = ds[0]
    assert x.shape == (1, N_CHANNELS, N_TIMES), f"bad epoch shape {x.shape}"
    assert y.dtype == torch.long

    weights = ds.class_weights()
    assert weights.shape == (N_CLASSES,)
    assert abs(float(weights.mean()) - 1.0) < 0.01, "class weights should average ~1"

    pids = ds.participant_indices("P001")
    assert len(pids) == N_CLASSES * EPOCHS_PER_CLASS
    print(f"  OK  len={len(ds)}, epoch_shape={tuple(x.shape)}, "
          f"class_weights={weights.tolist()}")


# ---------------------------------------------------------------------------
# Stage 2: EEGNet forward + all freeze modes
# ---------------------------------------------------------------------------

def test_eegnet() -> None:
    print("\n[Stage 2] EEGNet forward + freeze modes ...")
    model = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES)
    x = torch.randn(8, 1, N_CHANNELS, N_TIMES)
    logits = model(x)
    assert logits.shape == (8, N_CLASSES)
    scores = model.overload_score(x)
    assert scores.shape == (8,)
    assert scores.min().item() >= 0.0 and scores.max().item() <= 1.0

    expected_trainable = {
        "none": 0,
        "head_only": sum(p.numel() for p in model.classifier.parameters()),
        "late_layers": sum(
            p.numel() for p in list(model.separable_block.parameters())
            + list(model.classifier.parameters())
        ),
        "full": sum(p.numel() for p in model.parameters()),
    }
    for mode, expected in expected_trainable.items():
        m = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES)
        m.set_freeze_mode(mode)
        actual = sum(p.numel() for p in m.parameters() if p.requires_grad)
        assert actual == expected, f"freeze_mode={mode}: expected {expected}, got {actual}"
    print(f"  OK  logits={tuple(logits.shape)}, all freeze modes correct")


# ---------------------------------------------------------------------------
# Stage 3: Pretraining (train_mwl_model.py via subprocess)
# ---------------------------------------------------------------------------

def test_pretraining(dataset_path: Path, model_dir: Path) -> None:
    print("\n[Stage 3] Pretraining (train_mwl_model.py, 3 epochs) ...")
    python = sys.executable
    script = str(_REPO_ROOT / "scripts" / "train_mwl_model.py")
    cmd = [
        python, script,
        "--dataset", str(dataset_path),
        "--out-dir", str(model_dir),
        "--epochs", "3",
        "--patience", "2",
        "--batch-size", "16",
        "--device", "cpu",
        "--seed", "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  STDOUT:\n" + result.stdout[-2000:])
        print("  STDERR:\n" + result.stderr[-2000:])
        raise RuntimeError(f"train_mwl_model.py exited {result.returncode}")

    weights = model_dir / "eegnet_pretrained.pt"
    log = model_dir / "eegnet_pretrained_log.json"
    assert weights.exists(), "pretrained weights not written"
    assert log.exists(), "training log not written"

    with open(log) as f:
        log_data = json.load(f)
    assert "best_epoch" in log_data
    assert len(log_data["history"]) >= 1
    print(f"  OK  best_epoch={log_data['best_epoch']}  "
          f"best_val_loss={log_data['best_val_loss']:.4f}")


# ---------------------------------------------------------------------------
# Stage 4: Fine-tuning + ROC (calibrate_participant.py internals with
# synthetic epochs — bypasses XDF loading, tests all ML code paths)
# ---------------------------------------------------------------------------

def test_calibration(pretrained_path: Path, model_dir: Path, rng: np.random.Generator) -> None:
    print("\n[Stage 4] Fine-tuning + ROC (calibrate_participant internals) ...")

    # Import calibrate_participant functions directly
    import calibrate_participant as cp

    device = torch.device("cpu")

    # Synthetic calibration data: 3 classes × EPOCHS_PER_CLASS epochs
    cal_epochs_list, cal_labels_list = [], []
    for label_idx in range(N_CLASSES):
        epochs = rng.standard_normal(
            (EPOCHS_PER_CLASS, N_CHANNELS, N_TIMES)
        ).astype(np.float32)
        labels = np.full(EPOCHS_PER_CLASS, label_idx, dtype=np.int64)
        cal_epochs_list.append(epochs)
        cal_labels_list.append(labels)

    epochs_all = np.concatenate(cal_epochs_list, axis=0)
    labels_all = np.concatenate(cal_labels_list, axis=0).astype(np.int64)

    # Per-class temporal split
    train_eps, train_lbs, test_eps, test_lbs = [], [], [], []
    for lvl_idx in range(N_CLASSES):
        mask = labels_all == lvl_idx
        e_tr, l_tr, e_te, l_te = cp._temporal_split(
            epochs_all[mask], labels_all[mask], test_frac=0.20
        )
        train_eps.append(e_tr); train_lbs.append(l_tr)
        test_eps.append(e_te);  test_lbs.append(l_te)

    train_epochs = np.concatenate(train_eps)
    train_labels = np.concatenate(train_lbs).astype(np.int64)
    test_epochs  = np.concatenate(test_eps)
    test_labels  = np.concatenate(test_lbs).astype(np.int64)

    # Early-stopping split within training data
    split = int(len(train_labels) * 0.85)
    ft_loader = cp._build_loaders(
        train_epochs[:split], train_labels[:split], batch_size=16, device=device
    )
    es_tensor = (train_epochs[split:], train_labels[split:])

    results = []
    for mode in cp.ALL_MODES:
        model = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES).to(device)
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state)

        if mode != "none":
            cp._finetune(
                model, mode, ft_loader, es_tensor,
                max_epochs=3, patience=2, device=device,
            )

        metrics = cp._evaluate(model, test_epochs, test_labels, device)
        with torch.no_grad():
            high_scores = model.predict_proba(
                torch.from_numpy(test_epochs).unsqueeze(1).to(device)
            ).cpu().numpy()[:, HIGH_CLASS]
        thresh, j = cp._youden_threshold(
            (test_labels == LABEL_MAP["HIGH"]).astype(int),
            high_scores,
        )

        # Basic sanity: metrics exist and are in plausible ranges
        assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
        assert 0.0 <= metrics["auc_high_vs_rest"] <= 1.0
        assert 0.0 <= thresh <= 1.0

        results.append((mode, metrics["balanced_accuracy"], metrics["auc_high_vs_rest"], thresh))
        print(f"  [{mode:<12}] bal_acc={metrics['balanced_accuracy']:.3f}  "
              f"AUC={metrics['auc_high_vs_rest']:.3f}  threshold={thresh:.3f}")

    # Save a minimal comparison table to results/ (mirrors calibrate_participant.py output)
    results_dir = _REPO_ROOT / "results" / "synthetic_test"
    results_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(results_dir / "adaptation_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "balanced_accuracy", "auc_high_vs_rest", "threshold_youden"])
        w.writerows(results)

    print(f"  OK  comparison table → results/synthetic_test/adaptation_comparison.csv")


# ---------------------------------------------------------------------------
# Stage 5: ROC / AUC helpers standalone
# ---------------------------------------------------------------------------

def test_roc_helpers() -> None:
    print("\n[Stage 5] ROC helper functions ...")
    import calibrate_participant as cp

    rng = np.random.default_rng(99)

    # Perfect classifier
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = cp._roc_auc(y, s)
    assert abs(auc - 1.0) < 0.01, f"perfect AUC should be ~1.0, got {auc}"

    # Random classifier
    y2 = rng.integers(0, 2, size=200)
    s2 = rng.uniform(0, 1, size=200)
    auc2 = cp._roc_auc(y2, s2)
    assert 0.3 < auc2 < 0.7, f"random AUC should be ~0.5, got {auc2}"

    thresh, j = cp._youden_threshold(y, s)
    assert 0.0 <= thresh <= 1.0
    assert j > 0.0

    print(f"  OK  perfect_AUC={auc:.3f}  random_AUC={auc2:.3f}  "
          f"youden_thresh={thresh:.2f}  youden_J={j:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Synthetic end-to-end pipeline test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dataset_path = tmp_path / "dataset.h5"
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Write synthetic HDF5
        print("\n[Setup] Writing synthetic HDF5 dataset ...")
        _write_synthetic_hdf5(dataset_path, ["P001", "P002"], rng)

        for name, fn, kwargs in [
            ("Dataset round-trip",    test_dataset,    {"dataset_path": dataset_path}),
            ("EEGNet forward+freeze", test_eegnet,     {}),
            ("Pretraining",           test_pretraining, {"dataset_path": dataset_path, "model_dir": model_dir}),
            ("Fine-tuning + ROC",     test_calibration, {
                "pretrained_path": model_dir / "eegnet_pretrained.pt",
                "model_dir": model_dir,
                "rng": rng,
            }),
            ("ROC helpers",           test_roc_helpers, {}),
        ]:
            try:
                fn(**kwargs)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED stages: {failures}")
        sys.exit(1)
    else:
        print("All stages passed.")
        print("=" * 60)


if __name__ == "__main__":
    main()
