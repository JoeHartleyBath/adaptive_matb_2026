"""Leave-one-out cross-validation: pretrain + fine-tune for every participant.

Runs all 44 LOO folds sequentially.  For each held-out participant:
  1. Pretrain EEGNet on the remaining 43 participants  (train_mwl_model.py)
  2. Fine-tune on that participant across 4 freeze modes (test_pretrain_pipeline.py)
  3. Save per-fold results

Final summary JSON written to:
    {loo_root}/loo_cv_summary.json

Usage (from repo root with VR-TSST venv):
    C:\\vr_tsst_2025\\.venv\\Scripts\\python.exe scripts/run_loo_cv.py [options]

    --dataset PATH    Path to continuous data directory
    --loo-dir PATH    Root output directory (default: {vr_tsst}/output/matb_pretrain/loo)
    --exclude P ..    QC-failure PIDs to skip entirely (default: P02 P08 P46)
    --dry-run         Print commands without executing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from ml.pretrain_loader import PretrainDataDir  # noqa: E402

_PYTHON = r"C:\vr_tsst_2025\.venv\Scripts\python.exe"
_DEFAULT_DATASET = Path("C:/vr_tsst_2025/output/matb_pretrain/continuous")
_DEFAULT_LOO_DIR = Path("C:/vr_tsst_2025/output/matb_pretrain/loo")
_QC_CONFIG = _REPO_ROOT / "config" / "pretrain_qc.yaml"


def _load_exclusions() -> list[str]:
    """Read excluded PIDs from config/pretrain_qc.yaml."""
    if not _QC_CONFIG.exists():
        print(f"[WARN] QC config not found: {_QC_CONFIG} — using empty exclusion list", file=sys.stderr)
        return []
    cfg = yaml.safe_load(_QC_CONFIG.read_text())
    return list(cfg.get("excluded_participants", {}).keys())


_DEFAULT_EXCLUDE = _load_exclusions()  # read at import time; overridable via --exclude


def _run(cmd: list[str], dry_run: bool) -> int:
    print("\n>>> " + " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--loo-dir", type=Path, default=_DEFAULT_LOO_DIR)
    parser.add_argument("--exclude", nargs="*", default=_DEFAULT_EXCLUDE, metavar="PID")
    parser.add_argument(
        "--participants", nargs="*", default=None, metavar="PID",
        help="Run only these PIDs (subset of eligible folds). Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exclude_set = set(args.exclude)

    # Read all PIDs from continuous data directory
    data_dir = PretrainDataDir(args.dataset)
    all_pids: list[str] = data_dir.available_pids()

    folds = [p for p in all_pids if p not in exclude_set]

    # Optional subset filter
    if args.participants is not None:
        requested = set(args.participants)
        missing = requested - set(folds)
        if missing:
            print(f"[WARN] PIDs not in eligible folds (excluded or absent): {sorted(missing)}",
                  file=sys.stderr)
        folds = [p for p in folds if p in requested]

    print(f"Dataset : {args.dataset}")
    print(f"Folds   : {len(folds)}  ({', '.join(folds)})")
    print(f"Excluded: {sorted(exclude_set)}")
    print(f"Output  : {args.loo_dir}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    summary: list[dict] = []
    t_start_total = time.time()

    for i, pid in enumerate(folds, 1):
        print(f"\n{'='*60}")
        print(f"Fold {i}/{len(folds)} — hold-out: {pid}")
        print(f"{'='*60}")
        t_fold = time.time()

        fold_dir = args.loo_dir / pid

        # ---- Step 1: pretrain ----
        pretrain_cmd = [
            _PYTHON, "scripts/train_mwl_model.py",
            "--dataset", str(args.dataset),
            "--out-dir", str(fold_dir),
            "--hold-out", pid,
            "--exclude", *sorted(exclude_set),
        ]
        rc = _run(pretrain_cmd, args.dry_run)
        if rc != 0:
            print(f"[WARN] Pretrain failed for {pid} (exit {rc}) — skipping fine-tune")
            summary.append({"pid": pid, "pretrain_status": "failed", "finetune": None})
            continue

        # ---- Step 2: fine-tune ----
        pretrained_pt = fold_dir / "eegnet_pretrained.pt"
        channel_stats_pt = fold_dir / "channel_stats.npz"
        finetune_cmd = [
            _PYTHON, "scripts/test_pretrain_pipeline.py",
            "--pretrained", str(pretrained_pt),
            "--dataset", str(args.dataset),
            "--test-pid", pid,
            "--channel-stats", str(channel_stats_pt),
        ]
        rc = _run(finetune_cmd, args.dry_run)
        if rc != 0:
            print(f"[WARN] Fine-tune failed for {pid} (exit {rc})")
            summary.append({"pid": pid, "pretrain_status": "ok", "finetune": None})
            continue

        # ---- collect results ----
        results_json = _REPO_ROOT / "results" / "test_pretrain" / "results.json"
        finetune_results = None
        if results_json.exists() and not args.dry_run:
            with open(results_json) as fp:
                data = json.load(fp)
            # Flatten list [{mode, auc, bal_acc}, ...] → {mode: {auc, bal_acc}}
            finetune_results = {
                r["mode"]: {"auc": r["auc"], "bal_acc": r["bal_acc"]}
                for r in data.get("results", [])
            }
            # Copy fold results alongside the pretrained model for provenance
            fold_results_path = fold_dir / "finetune_results.json"
            with open(fold_results_path, "w") as fp:
                json.dump(data, fp, indent=2)

        elapsed = time.time() - t_fold
        print(f"\nFold {pid} done in {elapsed/60:.1f} min")
        summary.append({
            "pid": pid,
            "pretrain_status": "ok",
            "finetune": finetune_results,
            "elapsed_s": round(elapsed),
        })

    # ---- write summary ----
    total_elapsed = time.time() - t_start_total
    summary_out = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "n_folds": len(folds),
        "excluded": sorted(exclude_set),
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "folds": summary,
    }

    # Aggregate per-mode AUC across folds
    for mode in ("none", "head_only", "late_layers", "full"):
        aucs = [
            fold["finetune"][mode]["auc"]
            for fold in summary
            if fold.get("finetune") and mode in fold["finetune"]
        ]
        if aucs:
            import numpy as np
            summary_out[f"mean_auc_{mode}"] = round(float(np.mean(aucs)), 4)
            summary_out[f"std_auc_{mode}"] = round(float(np.std(aucs)), 4)
            print(f"\n{mode:15s}  mean AUC = {summary_out[f'mean_auc_{mode}']:.4f}  "
                  f"± {summary_out[f'std_auc_{mode}']:.4f}  (n={len(aucs)})")

    args.loo_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.loo_dir / "loo_cv_summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary_out, fp, indent=2)

    print(f"\nTotal time : {total_elapsed/60:.1f} min")
    print(f"Summary    : {summary_path}")


if __name__ == "__main__":
    main()
