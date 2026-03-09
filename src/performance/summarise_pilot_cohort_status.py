"""Summarise pilot cohort status (external-only) for quick readiness decisions.

This script scans the external output root (default: C:/data/adaptive_matb) for
run manifests and per-run pilot results summaries, then writes two CSVs:

1) pilot_cohort_blocks_latest.csv
   - one row per block (practice + calibration)
   - intended for quick inspection and plotting

2) pilot_cohort_status_latest.csv
   - one row per check (workload manipulation + practice readiness)
   - traffic-light style status plus simple effect summaries

Design goals:
- Small-N robust checks (pilot-friendly)
- Pure stdlib
- External-only outputs (no writing into git)

Inputs:
- <output_root>/openmatb/<Pxxx>/<Sxxx>/run_manifest_*.json
- <session_root>/derived/pilot_results_summary_*.csv (preferred)
  (If missing, script can regenerate via export_pilot_performance_table.)

Usage:
  python src/python/summarise_pilot_cohort_status.py
  python src/python/summarise_pilot_cohort_status.py --output-root C:/data/adaptive_matb
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any, Optional


LEVEL_ORDER = {"LOW": 1, "MODERATE": 2, "HIGH": 3}


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and not x.strip():
        return None
    try:
        return float(x)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _latest_run_manifests(output_root: Path) -> list[Path]:
    """Return latest run_manifest per (participant, session)."""
    openmatb_root = output_root / "openmatb"
    if not openmatb_root.exists():
        return []

    latest: dict[tuple[str, str], Path] = {}

    for participant_dir in sorted(p for p in openmatb_root.iterdir() if p.is_dir()):
        if not participant_dir.name.upper().startswith("P"):
            continue
        for session_dir in sorted(s for s in participant_dir.iterdir() if s.is_dir()):
            manifests = sorted(session_dir.glob("run_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not manifests:
                continue
            latest[(participant_dir.name, session_dir.name)] = manifests[0]

    return sorted(latest.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_json(path: Path) -> dict:
    import json

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _find_or_build_results_csv(run_manifest: dict, run_manifest_path: Path, *, output_root: Path) -> Optional[Path]:
    session_root = Path(str(run_manifest.get("openmatb", {}).get("session_root", "")))
    if not session_root.exists():
        # Fallback to expected structure
        participant = str(run_manifest.get("participant") or "")
        session = str(run_manifest.get("session") or "")
        session_root = output_root / "openmatb" / participant / session

    derived = session_root / "derived"
    latest = derived / "pilot_results_summary_latest.csv"
    if latest.exists():
        return latest

    candidates = sorted(derived.glob("pilot_results_summary_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]

    # Older path: run_manifest.json.performance_table.csv
    legacy = run_manifest_path.with_suffix(run_manifest_path.suffix + ".performance_table.csv")
    if legacy.exists():
        return legacy

    # Regenerate if possible
    try:
        import export_pilot_performance_table as exporter

        out = exporter.export_table(run_manifest_path=run_manifest_path, out_csv=None)
        return out if out.exists() else None
    except Exception:
        return None


def _linear_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return num / denom


def _binomial_two_sided_p(successes: int, n: int) -> Optional[float]:
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        return None

    def pmf(k: int) -> float:
        return math.comb(n, k) / (2 ** n)

    lower = sum(pmf(k) for k in range(0, successes + 1))
    upper = sum(pmf(k) for k in range(successes, n + 1))
    p = 2 * min(lower, upper)
    return min(1.0, p)


def _traffic_light(frac_ok: Optional[float], n: int, *, hard_red: bool = False) -> str:
    if hard_red:
        return "RED"
    if frac_ok is None or n == 0:
        return "YELLOW"
    if n >= 2 and frac_ok >= 0.8:
        return "GREEN"
    if frac_ok >= 0.5:
        return "YELLOW"
    return "RED"


def summarise(output_root: Path, *, out_dir: Optional[Path] = None) -> tuple[Path, Path]:
    manifests = _latest_run_manifests(output_root)

    cohort_block_rows: list[dict[str, Any]] = []

    # Collect block-level rows across runs
    for mp in manifests:
        run_manifest = _load_json(mp)
        results_csv = _find_or_build_results_csv(run_manifest, mp, output_root=output_root)
        if results_csv is None or not results_csv.exists():
            continue

        rows = _read_csv_rows(results_csv)
        for r in rows:
            # Attach source pointers and QC context
            r_out: dict[str, Any] = dict(r)
            r_out["run_manifest_path"] = str(mp)
            r_out["results_csv_path"] = str(results_csv)
            cohort_block_rows.append(r_out)

    # Write block table
    if out_dir is None:
        out_dir = output_root / "openmatb" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks_path = out_dir / "pilot_cohort_blocks_latest.csv"
    if cohort_block_rows:
        fieldnames = list(cohort_block_rows[0].keys())
        _write_csv(blocks_path, cohort_block_rows, fieldnames)
    else:
        _write_csv(blocks_path, [], ["note"])

    # --- Status checks ---
    # Group calibration blocks by participant+session
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in cohort_block_rows:
        pid = str(r.get("participant") or "")
        sid = str(r.get("session") or "")
        by_run.setdefault((pid, sid), []).append(r)

    def collect_level_series(run_rows: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for rr in run_rows:
            if str(rr.get("scenario_type")) != "calibration":
                continue
            lvl = str(rr.get("workload_level") or "")
            if lvl not in LEVEL_ORDER:
                continue
            y = _safe_float(rr.get(metric))
            if y is None:
                continue
            xs.append(float(LEVEL_ORDER[lvl]))
            ys.append(float(y))
        return xs, ys

    workload_checks = [
        # metric, expected_direction (+1 means increase with level; -1 means decrease)
        ("tlx_mean", +1),
        ("observed_sysmon_failures", +1),
        ("observed_comms_radioprompts", +1),
        ("observed_resman_pump_failures", +1),
        ("tracking_center_deviation_rmse", +1),
        ("sysmon_accuracy", -1),
        ("comms_accuracy", -1),
        ("comms_response_time_mean", +1),
        ("resman_tolerance_rate", -1),
    ]

    status_rows: list[dict[str, Any]] = []

    for metric, expected in workload_checks:
        slopes: list[float] = []
        deltas: list[float] = []
        ok = 0
        n = 0

        for run_key, run_rows in by_run.items():
            xs, ys = collect_level_series(run_rows, metric)
            if len(xs) < 3:
                continue
            # Align by level (ensure we use LOW/MOD/HIGH once each)
            series = {}
            for x, y in zip(xs, ys):
                series[int(x)] = y
            if not all(k in series for k in (1, 2, 3)):
                continue
            ys_ord = [series[1], series[2], series[3]]
            slope = _linear_slope([1.0, 2.0, 3.0], ys_ord)
            if slope is None:
                continue
            slopes.append(float(slope))
            delta = float(ys_ord[2] - ys_ord[0])
            deltas.append(delta)

            n += 1
            if expected > 0 and slope > 0:
                ok += 1
            if expected < 0 and slope < 0:
                ok += 1

        frac_ok = (ok / n) if n else None
        p = _binomial_two_sided_p(ok, n) if n else None
        median_slope = statistics.median(slopes) if slopes else None
        median_delta = statistics.median(deltas) if deltas else None

        status_rows.append(
            {
                "check": "workload_monotonic_direction",
                "metric": metric,
                "expected": "increase" if expected > 0 else "decrease",
                "n_runs": n,
                "n_in_expected_direction": ok,
                "frac_in_expected_direction": frac_ok,
                "binomial_p_two_sided": p,
                "median_slope": median_slope,
                "median_delta_high_minus_low": median_delta,
                "status": _traffic_light(frac_ok, n),
            }
        )

    # Practice readiness: conservative sanity checks (not inferential)
    practice_fail = 0
    practice_n = 0
    for run_key, run_rows in by_run.items():
        practice_blocks = [r for r in run_rows if str(r.get("scenario_type")) in {"practice", "practice_intro"}]
        if not practice_blocks:
            continue
        practice_n += 1

        hard_fail = False
        for r in practice_blocks:
            if str(r.get("abort_reason") or "").strip():
                hard_fail = True
                break
            # Ensure demands actually occurred (scenario ran)
            for k in ("observed_sysmon_failures", "observed_comms_radioprompts", "observed_resman_pump_failures"):
                v = _safe_float(r.get(k))
                if v is None:
                    hard_fail = True
                    break
            if hard_fail:
                break

        if hard_fail:
            practice_fail += 1

    frac_ok = ((practice_n - practice_fail) / practice_n) if practice_n else None
    status_rows.append(
        {
            "check": "practice_blocks_basic_readiness",
            "metric": "abort_reason + demand_counts_present",
            "expected": "all practice blocks complete without abort + demand counts present",
            "n_runs": practice_n,
            "n_in_expected_direction": (practice_n - practice_fail) if practice_n else 0,
            "frac_in_expected_direction": frac_ok,
            "binomial_p_two_sided": _binomial_two_sided_p(practice_n - practice_fail, practice_n) if practice_n else None,
            "median_slope": None,
            "median_delta_high_minus_low": None,
            "status": _traffic_light(frac_ok, practice_n, hard_red=(practice_fail > 0)),
        }
    )

    status_path = out_dir / "pilot_cohort_status_latest.csv"
    if status_rows:
        fieldnames = list(status_rows[0].keys())
        _write_csv(status_path, status_rows, fieldnames)
    else:
        _write_csv(status_path, [], ["note"])

    return blocks_path, status_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise pilot cohort readiness status (external-only).")
    parser.add_argument(
        "--output-root",
        default=None,
        help="External output root (default: C:/data/adaptive_matb)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for cohort tables (default: <output_root>/openmatb/derived)",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root else Path(r"C:\data\adaptive_matb")
    out_dir = Path(args.out_dir) if args.out_dir else None

    try:
        blocks, status = summarise(output_root, out_dir=out_dir)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Wrote cohort block table: {blocks}")
    print(f"Wrote cohort status table: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
