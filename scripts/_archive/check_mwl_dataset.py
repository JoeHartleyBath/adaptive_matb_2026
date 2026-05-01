"""Sanity-check the MWL pretraining dataset and cross-reference LOO-CV results.

Reads per-participant continuous HDF5 files (via PretrainDataDir) and completed
LOO fold results.  Does NOT import from src/ml/ and does NOT write anything the
LOO script depends on.  Safe to run while the LOO is in progress.

Usage
-----
    python scripts/check_mwl_dataset.py [--dataset PATH] [--loo-dir PATH]
    # or with the vr_tsst venv (has sklearn):
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/check_mwl_dataset.py

Outputs
-------
    results/figures/mwl_dataset_sanity/
        01_amplitude_rms.png         Per-participant epoch RMS boxplot
        02_label_ordering.png        Label sequence per participant
        03_bandpower_by_class.png    θ/α band-power difference HIGH-LOW
        04_logreg_auc.png            Per-participant logistic-regression AUC
        05_eegnet_vs_logreg_auc.png  EEGNet AUC vs logistic AUC scatter
    results/figures/mwl_dataset_sanity/integrity_table.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

_QC_CONFIG = Path(__file__).resolve().parent.parent / "config" / "pretrain_qc.yaml"
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import PretrainDataDir  # noqa: E402


def _load_default_exclusions() -> list[str]:
    if _YAML_OK and _QC_CONFIG.exists():
        cfg = _yaml.safe_load(_QC_CONFIG.read_text())
        return list(cfg.get("excluded_participants", {}).keys())
    return []

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_DATASET = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_DEFAULT_LOO_DIR = Path("C:/vr_tsst_2025/output/matb_pretrain/loo")
_FIG_DIR = _REPO_ROOT / "results" / "figures" / "mwl_dataset_sanity"

_SRATE = 128.0   # Hz — read from HDF5 attrs, this is the fallback
_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zscore(epoch: np.ndarray) -> np.ndarray:
    """Global z-score over all C×T — matches MwlDataset.__getitem__."""
    mu, sigma = epoch.mean(), epoch.std()
    return (epoch - mu) / sigma if sigma > 1e-8 else epoch


def _bandpower_welch(epochs: np.ndarray, srate: float) -> dict[str, np.ndarray]:
    """
    Compute mean log-band power per epoch across all channels.

    Parameters
    ----------
    epochs : (N, C, T) float32
    srate  : sampling rate in Hz

    Returns
    -------
    dict band_name -> (N,) float array of mean log-power
    """
    N, C, T = epochs.shape
    # Average across channels first (N, T) then Welch — reduces N*C calls to N.
    # Valid for this sanity check: we want the mean spectral profile per epoch.
    mean_signal = epochs.mean(axis=1)          # (N, T)
    freqs, psd = welch(mean_signal, fs=srate, nperseg=min(256, T), axis=-1)
    # psd shape: (N, F)

    results: dict[str, np.ndarray] = {}
    for band, (flo, fhi) in _BANDS.items():
        mask = (freqs >= flo) & (freqs < fhi)
        bp = psd[:, mask].mean(axis=1)
        results[band] = np.log10(bp + 1e-30)
    return results


def _load_participant(data_dir: PretrainDataDir, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (not z-scored) epochs (N, C, T) and labels (N,)."""
    epochs, labels, _ = data_dir.load_task_epochs(pid)
    return epochs, labels


def _remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map sparse {0, 2} → dense {0, 1}.  Pass-through if already dense."""
    unique = sorted(set(labels.tolist()))
    if unique == list(range(len(unique))):
        return labels
    lmap = {old: new for new, old in enumerate(unique)}
    return np.array([lmap[lb] for lb in labels], dtype=np.int64)


def _load_loo_results(loo_dir: Path) -> dict[str, dict[str, float]]:
    """
    Return {pid: {mode: auc, ...}, ...} for all completed folds.
    """
    results: dict[str, dict[str, float]] = {}
    for fold_dir in sorted(loo_dir.iterdir()):
        jpath = fold_dir / "finetune_results.json"
        if jpath.exists():
            data = json.loads(jpath.read_text())
            pid = fold_dir.name
            results[pid] = {r["mode"]: r["auc"] for r in data["results"]}
    return results


# ---------------------------------------------------------------------------
# Check 1 — Integrity table
# ---------------------------------------------------------------------------

def check_integrity(pids: list[str], data_dir: PretrainDataDir) -> list[dict]:
    print("\n=== Check 1: Integrity ===")
    rows = []
    header = f"{'PID':<6} {'N':>5} {'LOW':>5} {'HIGH':>5} {'ratio':>6} {'NaN%':>6} {'flat%':>6}"
    print(header)
    print("-" * len(header))

    for pid in pids:
        epochs, labels = _load_participant(data_dir, pid)
        labels_r = _remap_labels(labels)
        N = len(epochs)
        n_low  = int((labels_r == 0).sum())
        n_high = int((labels_r == 1).sum())
        ratio  = min(n_low, n_high) / max(n_low, n_high) if max(n_low, n_high) > 0 else 0.0
        nan_pct   = float(np.isnan(epochs).mean() * 100)
        flat_pct  = float(
            np.mean([ep.std() < 1e-6 for ep in epochs]) * 100
        )
        rows.append({
            "pid": pid, "N": N, "n_low": n_low, "n_high": n_high,
            "ratio": ratio, "nan_pct": nan_pct, "flat_pct": flat_pct,
        })
        flag = ""
        if ratio < 0.40:    flag += " [IMBALANCED]"
        if nan_pct > 3.0:   flag += " [NaN]"
        if flat_pct > 5.0:  flag += " [FLAT]"
        print(f"{pid:<6} {N:>5} {n_low:>5} {n_high:>5} {ratio:>6.3f} {nan_pct:>6.2f} {flat_pct:>6.2f}{flag}")

    return rows


# ---------------------------------------------------------------------------
# Check 2 — Amplitude RMS distribution
# ---------------------------------------------------------------------------

def check_amplitude(pids: list[str], data_dir: PretrainDataDir, fig_dir: Path) -> dict[str, float]:
    print("\n=== Check 2: Amplitude RMS (pre z-score) ===")
    rms_per_pid: dict[str, np.ndarray] = {}
    for pid in pids:
        epochs, _ = _load_participant(data_dir, pid)
        rms = np.sqrt((epochs ** 2).mean(axis=(1, 2)))   # (N,) global RMS per epoch
        rms_per_pid[pid] = rms

    median_rms = {pid: float(np.median(v)) for pid, v in rms_per_pid.items()}
    ordered = sorted(pids, key=lambda p: median_rms[p])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot(
        [rms_per_pid[p] for p in ordered],
        labels=ordered,
        showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="navy", linewidth=2),
    )
    ax.set_xlabel("Participant (sorted by median RMS)")
    ax.set_ylabel("Epoch global RMS (µV)")
    ax.set_title("Per-epoch amplitude RMS before z-score — cross-participant spread")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_amplitude_rms.png", dpi=150)
    plt.close(fig)
    print(f"  Median RMS range: {min(median_rms.values()):.2f} - {max(median_rms.values()):.2f}")
    print(f"  Saved: 01_amplitude_rms.png")
    # Persist so later checks can reload when --from-check skips this step
    import json as _json
    (fig_dir / "median_rms.json").write_text(_json.dumps(median_rms))
    return median_rms


# ---------------------------------------------------------------------------
# Check 3 — Label temporal ordering
# ---------------------------------------------------------------------------

def check_label_ordering(pids: list[str], data_dir: PretrainDataDir, fig_dir: Path) -> None:
    print("\n=== Check 3: Label temporal ordering ===")
    n = len(pids)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2))
    axes = axes.flatten()

    for i, pid in enumerate(pids):
        _, labels = _load_participant(data_dir, pid)
        labels_r = _remap_labels(labels)
        ax = axes[i]
        ax.scatter(range(len(labels_r)), labels_r, s=0.5, c=labels_r,
                   cmap="bwr", vmin=0, vmax=1, rasterized=True)
        ax.set_title(pid, fontsize=8)
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["L", "H"], fontsize=6)
        ax.set_xlabel("epoch idx", fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Label sequence per participant (blue=LOW, red=HIGH)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_label_ordering.png", dpi=150)
    plt.close(fig)
    print("  Saved: 02_label_ordering.png")


# ---------------------------------------------------------------------------
# Check 4 — Band-power difference HIGH − LOW
# ---------------------------------------------------------------------------

def check_bandpower(
    pids: list[str], data_dir: PretrainDataDir, fig_dir: Path, srate: float
) -> dict[str, dict[str, float]]:
    print("\n=== Check 4: Band-power difference (HIGH - LOW) ===")
    band_diffs: dict[str, dict[str, float]] = {}  # pid -> band -> diff

    for pid in pids:
        epochs, labels = _load_participant(data_dir, pid)
        labels_r = _remap_labels(labels)
        bp = _bandpower_welch(epochs, srate)
        diffs: dict[str, float] = {}
        for band, vals in bp.items():
            hi = vals[labels_r == 1].mean() if (labels_r == 1).any() else 0.0
            lo = vals[labels_r == 0].mean() if (labels_r == 0).any() else 0.0
            diffs[band] = float(hi - lo)
        band_diffs[pid] = diffs

    bands = list(_BANDS.keys())
    n = len(pids)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), sharey=False)
    axes = axes.flatten()
    colours = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]

    for i, pid in enumerate(pids):
        diffs = [band_diffs[pid][b] for b in bands]
        ax = axes[i]
        bars = ax.bar(bands, diffs, color=colours, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(pid, fontsize=8)
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        ax.tick_params(axis="y", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Log band-power difference HIGH − LOW (positive = more power at HIGH load)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_bandpower_by_class.png", dpi=150)
    plt.close(fig)
    print("  Saved: 03_bandpower_by_class.png")
    return band_diffs


# ---------------------------------------------------------------------------
# Check 5 — Logistic regression AUC (θ + α features)
# ---------------------------------------------------------------------------

def check_logreg_auc(
    pids: list[str], data_dir: PretrainDataDir, fig_dir: Path, srate: float
) -> dict[str, float]:
    print("\n=== Check 5: Logistic regression AUC (theta+alpha band features) ===")
    logreg_aucs: dict[str, float] = {}

    for pid in pids:
        epochs, labels = _load_participant(data_dir, pid)
        labels_r = _remap_labels(labels)
        bp = _bandpower_welch(epochs, srate)
        # Feature matrix: θ and α mean log-power per epoch (2 features)
        X = np.stack([bp["theta"], bp["alpha"]], axis=1)   # (N, 2)
        y = labels_r

        if len(np.unique(y)) < 2 or len(y) < 10:
            logreg_aucs[pid] = 0.5
            continue

        cv = StratifiedKFold(n_splits=5, shuffle=False)
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf.fit(X_tr, y[train_idx])
            prob = clf.predict_proba(X_te)[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y[test_idx], prob))
            except ValueError:
                fold_aucs.append(0.5)
        logreg_aucs[pid] = float(np.mean(fold_aucs))

    ordered = sorted(pids, key=lambda p: logreg_aucs[p])
    colours = ["#d73027" if logreg_aucs[p] < 0.5 else "#4575b4" for p in ordered]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.barh(ordered, [logreg_aucs[p] for p in ordered], color=colours, alpha=0.7)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("AUC (5-fold CV, θ+α features)")
    ax.set_title("Per-participant logistic regression AUC — upper bound on simple spectral separability")
    ax.set_xlim(0.3, 0.9)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_logreg_auc.png", dpi=150)
    plt.close(fig)

    mean_lr = np.mean(list(logreg_aucs.values()))
    print(f"  Mean theta+alpha logistic AUC: {mean_lr:.4f}")
    print(f"  Saved: 04_logreg_auc.png")
    return logreg_aucs


# ---------------------------------------------------------------------------
# Check 6 — EEGNet AUC vs logistic AUC
# ---------------------------------------------------------------------------

def check_correlation(
    logreg_aucs: dict[str, float],
    median_rms: dict[str, float],
    loo_results: dict[str, dict[str, float]],
    fig_dir: Path,
) -> None:
    print("\n=== Check 6: EEGNet AUC vs logistic AUC ===")

    common_pids = sorted(set(logreg_aucs) & set(loo_results))
    if len(common_pids) < 5:
        print(f"  Only {len(common_pids)} completed folds available - skipping scatter.")
        return

    lr_aucs = np.array([logreg_aucs[p] for p in common_pids])
    eg_aucs = np.array([loo_results[p]["full"] for p in common_pids])
    rms_vals = np.array([median_rms.get(p, np.nan) for p in common_pids])

    # Pearson r
    r_lr  = float(np.corrcoef(lr_aucs, eg_aucs)[0, 1])
    r_rms = float(np.corrcoef(rms_vals, eg_aucs)[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- scatter 1: logistic AUC vs EEGNet AUC ---
    ax1.scatter(lr_aucs, eg_aucs, s=60, alpha=0.7, edgecolors="black", linewidths=0.5)
    for pid, x, y in zip(common_pids, lr_aucs, eg_aucs):
        ax1.annotate(pid, (x, y), fontsize=6, ha="left", va="bottom",
                     xytext=(2, 2), textcoords="offset points")
    # regression line
    if len(common_pids) >= 3:
        m, b = np.polyfit(lr_aucs, eg_aucs, 1)
        xs = np.linspace(lr_aucs.min(), lr_aucs.max(), 50)
        ax1.plot(xs, m * xs + b, "r--", alpha=0.6, linewidth=1)
    ax1.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax1.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax1.set_xlabel("θ+α logistic AUC")
    ax1.set_ylabel("EEGNet full-mode AUC (LOO)")
    ax1.set_title(f"Logistic AUC vs EEGNet AUC  (r = {r_lr:.3f})")

    # --- scatter 2: amplitude RMS vs EEGNet AUC ---
    valid = ~np.isnan(rms_vals)
    ax2.scatter(rms_vals[valid], eg_aucs[valid], s=60, alpha=0.7,
                edgecolors="black", linewidths=0.5)
    for pid, x, y in zip(
        [p for p, v in zip(common_pids, valid) if v],
        rms_vals[valid], eg_aucs[valid]
    ):
        ax2.annotate(pid, (x, y), fontsize=6, ha="left", va="bottom",
                     xytext=(2, 2), textcoords="offset points")
    if valid.sum() >= 3:
        m, b = np.polyfit(rms_vals[valid], eg_aucs[valid], 1)
        xs = np.linspace(rms_vals[valid].min(), rms_vals[valid].max(), 50)
        ax2.plot(xs, m * xs + b, "r--", alpha=0.6, linewidth=1)
    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Median epoch RMS (µV)")
    ax2.set_ylabel("EEGNet full-mode AUC (LOO)")
    ax2.set_title(f"Amplitude vs EEGNet AUC  (r = {r_rms:.3f})")

    fig.tight_layout()
    fig.savefig(fig_dir / "05_eegnet_vs_logreg_auc.png", dpi=150)
    plt.close(fig)

    print(f"  Participants with completed LOO folds: {len(common_pids)}")
    print(f"  Pearson r (logistic AUC vs EEGNet AUC): {r_lr:.3f}")
    print(f"  Pearson r (RMS vs EEGNet AUC):          {r_rms:.3f}")
    print(f"  Saved: 05_eegnet_vs_logreg_auc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=_DEFAULT_DATASET,
        help=f"Path to continuous data directory (default: {_DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--loo-dir", type=Path, default=_DEFAULT_LOO_DIR,
        help=f"LOO output directory (default: {_DEFAULT_LOO_DIR})",
    )
    parser.add_argument(
        "--from-check", type=int, default=1, metavar="N",
        help="Start from check N (1-6). Skip earlier checks. Default: 1.",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=None, metavar="PID",
        help="PIDs to exclude (default: read from config/pretrain_qc.yaml).",
    )
    args = parser.parse_args()

    exclude = set(args.exclude if args.exclude is not None else _load_default_exclusions())

    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset : {args.dataset}")
    print(f"LOO dir : {args.loo_dir}")
    print(f"Figures : {_FIG_DIR}")

    # --- open continuous data directory ---
    data_dir = PretrainDataDir(args.dataset)
    srate = data_dir.srate()
    pids = data_dir.available_pids()

    if exclude:
        pids_all = pids
        pids = [p for p in pids if p not in exclude]
        print(f"Excluded {len(pids_all) - len(pids)} participants: {sorted(exclude)}")

    print(f"\n{len(pids)} participants after exclusions: {', '.join(pids)}")

    # --- load LOO results (may be partial) ---
    loo_results = _load_loo_results(args.loo_dir) if args.loo_dir.exists() else {}
    print(f"{len(loo_results)} completed LOO folds found.")

    # --- run checks ---
    integrity_rows = check_integrity(pids, data_dir)         if args.from_check <= 1 else None
    if args.from_check <= 2:
        median_rms = check_amplitude(pids, data_dir, _FIG_DIR)
    else:
        _rms_cache = _FIG_DIR / "median_rms.json"
        if _rms_cache.exists():
            import json as _json
            median_rms = _json.loads(_rms_cache.read_text())
            print("\n=== Check 2: Amplitude RMS (skipped - loaded from cache) ===")
        else:
            print("\n=== Check 2: Amplitude RMS (skipped - no cache found, RMS scatter will be empty) ===")
            median_rms = {p: float("nan") for p in pids}
    if args.from_check <= 3: check_label_ordering(pids, data_dir, _FIG_DIR)
    band_diffs     = check_bandpower(pids, data_dir, _FIG_DIR, srate) if args.from_check <= 4 else None
    logreg_aucs    = check_logreg_auc(pids, data_dir, _FIG_DIR, srate) if args.from_check <= 5 else None

    if args.from_check <= 6:
        check_correlation(logreg_aucs, median_rms, loo_results, _FIG_DIR)

    # --- save integrity table as CSV ---
    if integrity_rows is not None:
        import csv
        csv_path = _FIG_DIR / "integrity_table.csv"
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=integrity_rows[0].keys())
            writer.writeheader()
            writer.writerows(integrity_rows)
        print(f"\n  Integrity table saved: {csv_path}")

    # --- flag summary ---
    if integrity_rows is not None:
        print("\n=== Summary: flagged participants ===")
        flagged = False
        for row in integrity_rows:
            reasons = []
            if row["ratio"] < 0.40:   reasons.append(f"imbalanced (ratio={row['ratio']:.2f})")
            if row["nan_pct"] > 3.0:  reasons.append(f"NaN ({row['nan_pct']:.1f}%)")
            if row["flat_pct"] > 5.0: reasons.append(f"flat epochs ({row['flat_pct']:.1f}%)")
            pid = row["pid"]
            if pid in loo_results and loo_results[pid].get("none", 1.0) < 0.45:
                reasons.append(f"EEGNet none-AUC={loo_results[pid]['none']:.3f}")
            if reasons:
                print(f"  {pid}: {', '.join(reasons)}")
                flagged = True
        if not flagged:
            print("  None - all participants within expected bounds.")

    print("\nDone.")


if __name__ == "__main__":
    main()
