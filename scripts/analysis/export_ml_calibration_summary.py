"""Export a per-participant ML calibration summary CSV.

Scans the external models directory for per-participant calibration artefacts
written by ``scripts/session/calibrate_participant.py`` and flattens them into
a single human-readable CSV suitable for cohort-level quality review.

Inputs (external, never committed to git):
    {output_root}/models/{PID}/model_config.json
    {output_root}/models/{PID}/confusion_matrix_{pid_lower}.json

Output (in-repo derived summary, alongside qc_audit.csv):
    results/ml_calibration_summary.csv
    results/ml_cohort_confusion_matrix.json

One row per participant.  Two summary rows appended at the bottom:
    _mean    — column-wise mean across all participants with valid data
    _median  — column-wise median

The cohort confusion matrix JSON sums the raw 10-fold CV confusion matrices
across all participants with valid data, and reports per-class recall and
overall accuracy derived from the summed counts.

Usage
-----
    python scripts/analysis/export_ml_calibration_summary.py
    python scripts/analysis/export_ml_calibration_summary.py \\
        --output-root D:/data/adaptive_matb \\
        --out results/ml_calibration_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Repo root (two levels up from this file: scripts/analysis/ -> scripts/ -> repo)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_ROOT = Path(r"C:\data\adaptive_matb")
_DEFAULT_OUT = _REPO_ROOT / "results" / "ml_calibration_summary.csv"
_DEFAULT_CM_OUT = _REPO_ROOT / "results" / "ml_cohort_confusion_matrix.json"
_DEFAULT_FIG_OUT = _REPO_ROOT / "results" / "figures" / "ml_cohort_confusion_matrix.png"

_CM_LABELS = ["LOW", "MODERATE", "HIGH"]

# Cohort label order for display / summary row grouping.
_COHORT_ORDER = ["dry_run", "self", "pilot", "study"]


def _cohort_label(pid: str) -> str:
    """Derive cohort from PID naming convention.

    PDRY*   -> dry_run
    PSELF*  -> self
    PPILOT* -> pilot
    P + digits -> study
    """
    p = pid.upper()
    if p.startswith("PDRY"):
        return "dry_run"
    if p.startswith("PSELF"):
        return "self"
    if p.startswith("PPILOT"):
        return "pilot"
    return "study"


# Column order in the output CSV.
FIELDNAMES: list[str] = [
    "pid",
    "cohort",
    "calibrated_at",
    "n_windows",
    "n_classes",
    "model_k",
    "threshold_method",
    # Deployed threshold
    "youden_threshold",
    "youdens_j",
    # Training-set threshold
    "train_youden_threshold",
    "train_youdens_j",
    # 10-fold CV threshold
    "kfold_youdens_j",
    "kfold_n_splits",
    # LORO threshold (leave-one-run-out; populated when ≥2 cal runs available)
    "loro_youdens_j",
    "loro_n_folds",
    # Accuracy
    "train_acc",
    "cv_10fold_acc",
    # Per-class recall (10-fold CV)
    "recall_LOW",
    "recall_MODERATE",
    "recall_HIGH",
]

# Numeric fields (used for summary rows).
_NUMERIC_FIELDS: list[str] = [f for f in FIELDNAMES if f not in (
    "pid", "cohort", "calibrated_at", "threshold_method",
)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and not x.strip():
        return None
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Cohort confusion matrix
# ---------------------------------------------------------------------------

def _read_cm_raw(cm_path: Path) -> Optional[list[list[int]]]:
    """Return the 3×3 cv_10fold confusion matrix as list-of-lists, or None."""
    cm = _load_json(cm_path)
    if cm is None:
        return None
    matrix = cm.get("cv_10fold", {}).get("confusion_matrix")
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or not all(isinstance(row, list) and len(row) == 3 for row in matrix)
    ):
        return None
    return matrix


def _sum_matrices(matrices: list[list[list[int]]]) -> list[list[int]]:
    """Element-wise sum of a list of 3×3 matrices."""
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for m in matrices:
        for i in range(3):
            for j in range(3):
                result[i][j] += m[i][j]
    return result


def _cohort_cm_stats(
    summed: list[list[int]],
    labels: list[str],
    n_participants: int,
    cohort_breakdown: str = "",
) -> dict:
    """Derive accuracy and per-class recall from summed confusion matrix."""
    total_correct = sum(summed[i][i] for i in range(3))
    total = sum(summed[i][j] for i in range(3) for j in range(3))
    accuracy = total_correct / total if total else None

    per_class: dict[str, Any] = {}
    for i, label in enumerate(labels):
        row_total = sum(summed[i])
        correct = summed[i][i]
        per_class[label] = {
            "n": row_total,
            "correct": correct,
            "recall": round(correct / row_total, 4) if row_total else None,
        }

    return {
        "n_participants": n_participants,
        "cohort_breakdown": cohort_breakdown,
        "labels": labels,
        "cv_10fold": {
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "confusion_matrix": summed,
            "per_class": per_class,
        },
    }


def _write_cohort_cm(stats: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    tmp.replace(out_path)


def _plot_cohort_cm(stats: dict, fig_path: Path) -> None:
    """Save a normalised heatmap of the cohort confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  WARNING: matplotlib not available — figure not saved.")
        return

    labels = stats["labels"]
    cv = stats["cv_10fold"]
    matrix = cv["confusion_matrix"]
    n_participants = stats["n_participants"]
    per_class = cv["per_class"]
    n = len(labels)

    # Row-normalise (recall per true class)
    norm = []
    for i in range(n):
        row_total = sum(matrix[i])
        norm.append([matrix[i][j] / row_total if row_total else 0.0 for j in range(n)])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)

    # Annotate cells: normalised value on top, raw count below
    for i in range(n):
        for j in range(n):
            val = norm[i][j]
            count = matrix[i][j]
            text_colour = "white" if val > 0.6 else "black"
            ax.text(
                j, i,
                f"{val:.2f}\n({count})",
                ha="center", va="center",
                fontsize=9, color=text_colour,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = cv.get("accuracy")
    acc_str = f"{acc:.3f}" if acc is not None else "N/A"
    breakdown = stats.get("cohort_breakdown", "")
    breakdown_str = f"\n{breakdown}" if breakdown else ""
    ax.set_title(
        f"Cohort confusion matrix (10-fold CV)\n"
        f"N={n_participants} participant(s)  |  overall acc={acc_str}{breakdown_str}"
    )

    fig.colorbar(im, ax=ax, label="Recall (row-normalised)")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved cohort CM figure to:\n  {fig_path}")


# ---------------------------------------------------------------------------
# Per-participant extraction
# ---------------------------------------------------------------------------

def _scan_model_dirs(output_root: Path) -> list[tuple[str, Path, Optional[Path]]]:
    """Return [(pid, model_config_path, confusion_matrix_path_or_None), ...].

    Handles two layouts:
      - flat:    models/{PID}/model_config.json
      - session: models/{PID}/{session}/model_config.json

    When a PID has multiple session subdirs the most-recently-modified
    model_config.json is used.

    Confusion matrix lookup order:
      1. {models_root}/{PID}/confusion_matrix_{pid_lower}.json  (flat)
      2. {models_root}/{PID}/**/confusion_matrix_{pid_lower}.json  (session)
      3. {_REPO_ROOT}/results/{pid_lower}/confusion_matrix_{pid_lower}.json
    """
    models_root = output_root / "models"
    if not models_root.exists():
        return []

    # Collect candidates: all model_config.json files up to 2 levels deep.
    # Key: top-level PID dir name.  Keep the most recently modified per PID.
    best: dict[str, Path] = {}
    for pid_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        pid = pid_dir.name
        # Skip non-participant directories (e.g. 'compare')
        if not pid.upper().startswith("P"):
            continue
        candidates = list(pid_dir.glob("model_config.json")) + \
                     list(pid_dir.glob("*/model_config.json"))
        if not candidates:
            continue
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        best[pid] = latest

    entries: list[tuple[str, Path, Optional[Path]]] = []
    for pid in sorted(best):
        mc_path = best[pid]
        pid_lower = pid.lower()
        cm_name = f"confusion_matrix_{pid_lower}.json"
        # Search in the same directory as model_config, then anywhere under
        # the PID models dir, then fall back to the in-repo results/ folder.
        pid_models_dir = models_root / pid
        cm_candidates = [
            mc_path.parent / cm_name,
            *pid_models_dir.glob(f"**/{cm_name}"),
            _REPO_ROOT / "results" / pid_lower / cm_name,
        ]
        cm_path = next((p for p in cm_candidates if p.exists()), None)
        entries.append((pid, mc_path, cm_path))

    return entries


def _read_participant_row(
    pid: str,
    mc_path: Path,
    cm_path: Optional[Path],
) -> dict[str, Any]:
    """Return a flat row dict for one participant.

    Missing values are represented as empty string so the CSV is still
    well-formed.
    """
    row: dict[str, Any] = {f: "" for f in FIELDNAMES}
    row["pid"] = pid
    row["cohort"] = _cohort_label(pid)

    # --- model_config.json ---
    mc = _load_json(mc_path)
    if mc is None:
        print(f"  WARNING: could not read {mc_path}")
    else:
        row["calibrated_at"]         = mc.get("calibrated_at", "")
        row["n_classes"]             = mc.get("n_classes", "")
        row["model_k"]               = mc.get("model_k", "")
        row["threshold_method"]      = mc.get("threshold_method", "")
        row["youden_threshold"]      = _fmt(mc.get("youden_threshold"))
        row["youdens_j"]             = _fmt(mc.get("youdens_j"))
        row["train_youden_threshold"] = _fmt(mc.get("train_youden_threshold"))
        row["train_youdens_j"]       = _fmt(mc.get("train_youdens_j"))
        row["kfold_youdens_j"]       = _fmt(mc.get("kfold_youdens_j"))
        row["kfold_n_splits"]        = mc.get("kfold_n_splits", "")
        row["loro_youdens_j"]        = _fmt(mc.get("loro_youdens_j"))
        row["loro_n_folds"]          = mc.get("loro_n_folds", "")

    # --- confusion_matrix_{pid}.json ---
    if cm_path is None:
        print(f"  INFO:    no confusion_matrix file for {pid} — accuracy columns left blank")
    else:
        cm = _load_json(cm_path)
        if cm is None:
            print(f"  WARNING: could not read {cm_path}")
        else:
            row["n_windows"] = cm.get("n_windows", "")

            ts = cm.get("training_set", {})
            row["train_acc"] = _fmt(ts.get("accuracy"))

            cv = cm.get("cv_10fold", {})
            row["cv_10fold_acc"] = _fmt(cv.get("accuracy"))

            per_class = cv.get("per_class", {})
            for level in ("LOW", "MODERATE", "HIGH"):
                cls_data = per_class.get(level, {})
                row[f"recall_{level}"] = _fmt(cls_data.get("recall"))

    return row


def _fmt(v: Any) -> Any:
    """Round floats to 4 d.p. for readability; pass other types through."""
    f = _safe_float(v)
    if f is None:
        return ""
    return round(f, 4)


# ---------------------------------------------------------------------------
# Summary rows
# ---------------------------------------------------------------------------

def _summary_rows(participant_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return mean+median rows grouped by cohort, then an overall block."""
    groups: dict[str, list[dict[str, Any]]] = {c: [] for c in _COHORT_ORDER}
    for row in participant_rows:
        groups.setdefault(row.get("cohort", "study"), []).append(row)

    def _stats_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        by_field: dict[str, list[float]] = {f: [] for f in _NUMERIC_FIELDS}
        for r in rows:
            for field in _NUMERIC_FIELDS:
                v = _safe_float(r.get(field))
                if v is not None:
                    by_field[field].append(v)

        def _make_row(pid_label: str, fn) -> dict[str, Any]:
            out: dict[str, Any] = {f: "" for f in FIELDNAMES}
            out["pid"] = pid_label
            for field in _NUMERIC_FIELDS:
                vals = by_field[field]
                if vals:
                    out[field] = round(fn(vals), 4)
            return out

        n = len(rows)
        result: list[dict[str, Any]] = []
        if n >= 1:
            result.append(_make_row(f"_mean_{label} (n={n})", lambda vs: sum(vs) / len(vs)))
        if n >= 2:
            result.append(_make_row(f"_median_{label} (n={n})", statistics.median))
        return result

    out: list[dict[str, Any]] = []
    for cohort in _COHORT_ORDER:
        rows = groups.get(cohort, [])
        if rows:
            out.extend(_stats_rows(rows, cohort))
    if len(participant_rows) >= 1:
        out.extend(_stats_rows(participant_rows, "all"))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def export_summary(
    output_root: Path,
    out_path: Path,
    cm_out_path: Path,
    fig_path: Path = _DEFAULT_FIG_OUT,
) -> Path:
    entries = _scan_model_dirs(output_root)
    if not entries:
        print(f"No model directories found under {output_root / 'models'}")
        return out_path

    participant_rows: list[dict[str, Any]] = []
    raw_matrices: dict[str, list[list[list[int]]]] = {}  # cohort -> matrices

    for pid, mc_path, cm_path in entries:
        print(f"  {pid}  ...", end=" ", flush=True)
        row = _read_participant_row(pid, mc_path, cm_path)
        participant_rows.append(row)
        if cm_path is not None:
            m = _read_cm_raw(cm_path)
            if m is not None:
                raw_matrices.setdefault(_cohort_label(pid), []).append(m)
        print("OK")

    all_rows = participant_rows + _summary_rows(participant_rows)
    _write_csv(out_path, all_rows, FIELDNAMES)
    print(f"\nWrote {len(participant_rows)} participant row(s) + summary to:\n  {out_path}")

    if raw_matrices:
        all_flat = [m for ms in raw_matrices.values() for m in ms]
        summed = _sum_matrices(all_flat)
        breakdown = ", ".join(
            f"{c}={len(raw_matrices[c])}" for c in _COHORT_ORDER if c in raw_matrices
        )
        stats = _cohort_cm_stats(
            summed, _CM_LABELS,
            n_participants=len(all_flat),
            cohort_breakdown=breakdown,
        )
        _write_cohort_cm(stats, cm_out_path)
        print(f"Wrote cohort confusion matrix ({len(all_flat)} participant(s): {breakdown}) to:\n  {cm_out_path}")
        _print_cohort_cm(stats)
        _plot_cohort_cm(stats, fig_path)
    else:
        print("No confusion matrix data found — cohort CM not written.")

    return out_path


def _print_cohort_cm(stats: dict) -> None:
    """Print a compact ASCII summary of the cohort confusion matrix."""
    labels = stats["labels"]
    cv = stats["cv_10fold"]
    matrix = cv["confusion_matrix"]
    per_class = cv["per_class"]
    n = stats["n_participants"]
    breakdown = stats.get("cohort_breakdown", "")
    breakdown_str = f" ({breakdown})" if breakdown else ""

    print(f"\n  Cohort confusion matrix (10-fold CV, N={n} participants{breakdown_str})")
    print(f"  Predicted →  {'   '.join(f'{l:>8}' for l in labels)}")
    for i, label in enumerate(labels):
        recall = per_class[label]["recall"]
        recall_str = f"{recall:.3f}" if recall is not None else "  N/A"
        row_str = "  ".join(f"{matrix[i][j]:>8d}" for j in range(3))
        print(f"  True {label:<10}{row_str}   recall={recall_str}")
    acc = cv["accuracy"]
    print(f"  Overall accuracy: {acc:.4f}" if acc is not None else "  Overall accuracy: N/A")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-participant ML calibration summary CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help=f"External data root (default: {_DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output CSV path (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--cm-out",
        type=Path,
        default=_DEFAULT_CM_OUT,
        help=f"Output cohort confusion matrix JSON path (default: {_DEFAULT_CM_OUT})",
    )
    parser.add_argument(
        "--fig-out",
        type=Path,
        default=_DEFAULT_FIG_OUT,
        help=f"Output figure path (default: {_DEFAULT_FIG_OUT})",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.output_root / 'models'}")
    print(f"Writing to: {args.out}")
    print(f"Cohort CM to: {args.cm_out}")
    print(f"Figure to:   {args.fig_out}\n")
    export_summary(args.output_root, args.out, args.cm_out, args.fig_out)


if __name__ == "__main__":
    main()
