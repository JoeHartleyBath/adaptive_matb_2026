"""Export a tidy per-scenario performance table (CSV) for pilot verification.

This script aggregates the per-scenario `*.performance_summary.json` artifacts
(created by `summarise_openmatb_performance.py`) into a single wide CSV.

Primary use-case: create a quick, human-auditable table for checking that the
workload manipulation (LOW/MODERATE/HIGH) behaved as expected, including:
- objective performance metrics (per task)
- subjective workload ratings (NASA-TLX via genericscales)
- practice blocks, to confirm practice effectiveness

Input is the run-level manifest written by `run_openmatb.py`:
  <output_root>/openmatb/<Pxxx>/<Sxxx>/run_manifest_*.json

Usage:
  python src/python/export_pilot_performance_table.py \
      --run-manifest C:/data/adaptive_matb/openmatb/P001/S001/run_manifest_20260206T105823.json

Notes:
- Pure stdlib.
- Output defaults to: <run_manifest>.performance_table.csv
- One row per scenario in the playlist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pick_main_segment_name(summary: dict[str, Any]) -> Optional[str]:
    segments = summary.get("segments")
    if not isinstance(segments, list):
        return None

    names = [s.get("name") for s in segments if isinstance(s, dict) and isinstance(s.get("name"), str)]
    for n in names:
        if "/calibration/" in n:
            return n
    for n in names:
        if "/TRAINING/" in n:
            return n
    return names[0] if names else None


def _pick_tlx_segment_name(summary: dict[str, Any]) -> Optional[str]:
    segments = summary.get("segments")
    if not isinstance(segments, list):
        return None

    for s in segments:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if isinstance(name, str) and "/TLX/" in name:
            return name
    return None


def _segment_duration_sec(summary: dict[str, Any], segment_name: str) -> Optional[float]:
    per = summary.get("per_segment")
    if not isinstance(per, dict):
        return None
    seg = per.get(segment_name)
    if not isinstance(seg, dict):
        return None
    window = seg.get("window")
    if not isinstance(window, dict):
        return None
    start = _safe_float(window.get("start_sec"))
    end = _safe_float(window.get("end_sec"))
    if start is None or end is None:
        return None
    if end < start:
        return None
    return float(end - start)


def _segment_window(summary: dict[str, Any], segment_name: str) -> Optional[tuple[float, float]]:
    per = summary.get("per_segment")
    if not isinstance(per, dict):
        return None
    seg = per.get(segment_name)
    if not isinstance(seg, dict):
        return None
    window = seg.get("window")
    if not isinstance(window, dict):
        return None
    start = _safe_float(window.get("start_sec"))
    end = _safe_float(window.get("end_sec"))
    if start is None or end is None:
        return None
    if end < start:
        return None
    return float(start), float(end)


def _count_resman_pump_failures(csv_path: Path, window: Optional[tuple[float, float]]) -> int:
    """Count resman pump-*-state failure events in the given time window."""

    start = window[0] if window else None
    end = window[1] if window else None

    count = 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip().lower()
            if row_type != "event":
                continue
            module = (row.get("module") or "").strip().lower()
            if module != "resman":
                continue
            address = (row.get("address") or "").strip().lower()
            if not (address.startswith("pump-") and address.endswith("-state")):
                continue
            value = (row.get("value") or "").strip().lower()
            if value != "failure":
                continue

            t = _safe_float(row.get("scenario_time"))
            if t is None:
                continue
            if start is not None and float(t) < float(start):
                continue
            if end is not None and float(t) > float(end):
                continue
            count += 1

    return count


def _count_demand_events(
    csv_path: Path,
    window: Optional[tuple[float, float]],
) -> dict[str, int]:
    """Count scheduled demand events in the event stream (not performance rows).

    This is a robust way to verify that the scenario schedule ran as intended.
    """

    start = window[0] if window else None
    end = window[1] if window else None

    counts = {
        "sysmon_failures": 0,
        "comms_radioprompts": 0,
        "resman_pump_failures": 0,
    }

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip().lower()
            if row_type != "event":
                continue

            t = _safe_float(row.get("scenario_time"))
            if t is None:
                continue
            if start is not None and float(t) < float(start):
                continue
            if end is not None and float(t) > float(end):
                continue

            module = (row.get("module") or "").strip().lower()
            address = (row.get("address") or "").strip().lower()
            value = (row.get("value") or "").strip().lower()

            if module == "sysmon" and "failure" in address and value in {"true", "1"}:
                counts["sysmon_failures"] += 1
                continue

            if module == "communications" and address == "radioprompt":
                counts["comms_radioprompts"] += 1
                continue

            if module == "resman" and address.startswith("pump-") and address.endswith("-state") and value == "failure":
                counts["resman_pump_failures"] += 1
                continue

    return counts


def _count_scheduled_demands_in_scenario(scenario_path: Path) -> dict[str, int]:
    """Count demand events scheduled in the scenario text artifact."""
    counts = {
        "sysmon_failures": 0,
        "comms_radioprompts": 0,
        "resman_pump_failures": 0,
    }
    for raw in scenario_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3:
            continue
        module = parts[1].lower()
        action = parts[2].lower()
        arg = (parts[3] if len(parts) >= 4 else "").strip().lower()

        if module == "sysmon" and "failure" in action and arg in {"true", "1"}:
            counts["sysmon_failures"] += 1
        elif module == "communications" and action == "radioprompt":
            counts["comms_radioprompts"] += 1
        elif module == "resman" and action.startswith("pump-") and action.endswith("-state") and arg == "failure":
            counts["resman_pump_failures"] += 1
    return counts


def _get_perf_root(summary: dict[str, Any], segment_name: Optional[str]) -> dict[str, Any]:
    """Return performance dict for a segment if present, else overall."""
    if segment_name:
        per = summary.get("per_segment")
        if isinstance(per, dict):
            seg = per.get(segment_name)
            if isinstance(seg, dict):
                perf = seg.get("performance")
                if isinstance(perf, dict):
                    return perf
    overall = summary.get("overall")
    return overall if isinstance(overall, dict) else {}


def _get_kpis_root(summary: dict[str, Any], segment_name: Optional[str]) -> dict[str, Any]:
    if segment_name:
        per = summary.get("per_segment")
        if isinstance(per, dict):
            seg = per.get(segment_name)
            if isinstance(seg, dict):
                kpis = seg.get("derived_kpis")
                if isinstance(kpis, dict):
                    return kpis
    kpis = summary.get("derived_kpis")
    return kpis if isinstance(kpis, dict) else {}


def _numeric_mean(perf: dict[str, Any], module: str, metric: str) -> Optional[float]:
    mod = perf.get(module)
    if not isinstance(mod, dict):
        return None
    m = mod.get(metric)
    if not isinstance(m, dict):
        return None
    numeric = m.get("numeric")
    if not isinstance(numeric, dict):
        return None
    return _safe_float(numeric.get("mean"))


def _bool_true_rate(perf: dict[str, Any], module: str, metric: str) -> tuple[Optional[float], int, int]:
    """Return (rate_true, true_count, false_count)."""
    mod = perf.get(module)
    if not isinstance(mod, dict):
        return None, 0, 0
    m = mod.get(metric)
    if not isinstance(m, dict):
        return None, 0, 0
    b = m.get("boolean")
    if not isinstance(b, dict):
        return None, 0, 0
    t = int(b.get("true") or 0)
    f = int(b.get("false") or 0)
    denom = t + f
    return ((t / denom) if denom else None), t, f


def _categorical_counts(perf: dict[str, Any], module: str, metric: str) -> dict[str, int]:
    mod = perf.get(module)
    if not isinstance(mod, dict):
        return {}
    m = mod.get(metric)
    if not isinstance(m, dict):
        return {}
    cat = m.get("categorical")
    if not isinstance(cat, dict):
        return {}
    out: dict[str, int] = {}
    for k, v in cat.items():
        if not isinstance(k, str):
            continue
        try:
            out[k] = int(v)
        except Exception:
            continue
    return out


def _scenario_type_and_level(scenario_name: str, main_segment: Optional[str]) -> tuple[str, Optional[str]]:
    name = scenario_name.lower()
    if "practice_intro" in name:
        return "practice_intro", None
    if "practice" in name:
        # Prefer the segment marker mapping if present.
        if isinstance(main_segment, str) and "/TRAINING/" in main_segment:
            if main_segment.endswith("/T1"):
                return "practice", "LOW"
            if main_segment.endswith("/T2"):
                return "practice", "MODERATE"
            if main_segment.endswith("/T3"):
                return "practice", "HIGH"
        if name.endswith("_low"):
            return "practice", "LOW"
        if name.endswith("_moderate"):
            return "practice", "MODERATE"
        if name.endswith("_high"):
            return "practice", "HIGH"
        return "practice", None

    if "calibration" in name:
        if name.endswith("_low"):
            return "calibration", "LOW"
        if name.endswith("_moderate"):
            return "calibration", "MODERATE"
        if name.endswith("_high"):
            return "calibration", "HIGH"
        return "calibration", None

    return "other", None


def _resman_tolerance_rate(perf: dict[str, Any]) -> tuple[Optional[float], int, int]:
    """Aggregate across all resman *_in_tolerance metrics."""
    mod = perf.get("resman")
    if not isinstance(mod, dict):
        return None, 0, 0

    true_total = 0
    false_total = 0

    for metric, payload in mod.items():
        if not isinstance(metric, str):
            continue
        if not metric.endswith("_in_tolerance"):
            continue
        if not isinstance(payload, dict):
            continue
        b = payload.get("boolean")
        if not isinstance(b, dict):
            continue
        true_total += int(b.get("true") or 0)
        false_total += int(b.get("false") or 0)

    denom = true_total + false_total
    return ((true_total / denom) if denom else None), true_total, false_total


def export_table(*, run_manifest_path: Path, out_csv: Optional[Path]) -> Path:
    run_manifest = _load_json(run_manifest_path)

    participant = str(run_manifest.get("participant") or "")
    session = str(run_manifest.get("session") or "")
    seq_id = str(run_manifest.get("seq_id") or "")
    created_at = str(run_manifest.get("created_at") or "")
    mode = run_manifest.get("mode") if isinstance(run_manifest.get("mode"), dict) else {}
    qc = run_manifest.get("qc") if isinstance(run_manifest.get("qc"), dict) else {}
    xdf_align = qc.get("xdf_alignment") if isinstance(qc.get("xdf_alignment"), dict) else {}

    repo_root = Path(__file__).resolve().parents[2]
    scenario_dir = repo_root / "scenarios"

    playlist = run_manifest.get("playlist")
    if not isinstance(playlist, list) or not playlist:
        raise ValueError("run manifest has no playlist")

    derived_dir = run_manifest_path.parent / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    def _default_out_path() -> Path:
        # Prefer a human-readable filename rather than the internal suffix.
        stem = run_manifest_path.stem
        if stem.startswith("run_manifest_"):
            tag = stem.replace("run_manifest_", "")
        else:
            tag = stem
        return derived_dir / f"pilot_results_summary_{tag}.csv"

    if out_csv is None:
        out_csv = _default_out_path()

    rows: list[dict[str, Any]] = []

    for block_index, item in enumerate(playlist):
        if not isinstance(item, dict):
            continue

        scenario_name = str(item.get("scenario_name") or "")
        scenario_filename = str(item.get("scenario_filename") or "")
        manifest_path = Path(str(item.get("manifest_path") or ""))

        abort_reason = None
        try:
            manifest = _load_json(manifest_path)
            abort_reason = manifest.get("abort_reason")
        except Exception:
            abort_reason = None

        summary_path = Path(str(manifest_path) + ".performance_summary.json")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing performance summary JSON: {summary_path}")

        summary = _load_json(summary_path)
        main_seg = _pick_main_segment_name(summary)
        tlx_seg = _pick_tlx_segment_name(summary)

        csv_path = Path(str(summary.get("csv_path") or ""))
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing session CSV referenced by summary: {csv_path}")

        perf_main = _get_perf_root(summary, main_seg)
        kpis_main = _get_kpis_root(summary, main_seg)

        scenario_type, level = _scenario_type_and_level(scenario_name, main_seg)

        # --- Derived KPIs ---
        tracking_rmse = None
        tracking_n = None
        tracking = kpis_main.get("tracking")
        if isinstance(tracking, dict):
            tracking_rmse = _safe_float(tracking.get("center_deviation_rmse"))
            try:
                tracking_n = int(tracking.get("n")) if tracking.get("n") is not None else None
            except Exception:
                tracking_n = None

        sysmon_acc = None
        sysmon_hits = None
        sysmon_miss = None
        sysmon_fa = None
        sysmon_total = None
        sysmon_demand_total = None
        sysmon = kpis_main.get("sysmon")
        if isinstance(sysmon, dict):
            sysmon_acc = _safe_float(sysmon.get("accuracy"))
            counts = sysmon.get("counts")
            if isinstance(counts, dict):
                try:
                    sysmon_hits = int(counts.get("HIT")) if counts.get("HIT") is not None else None
                    sysmon_miss = int(counts.get("MISS")) if counts.get("MISS") is not None else None
                    sysmon_fa = int(counts.get("FA")) if counts.get("FA") is not None else None
                    sysmon_total = int(counts.get("TOTAL")) if counts.get("TOTAL") is not None else None
                    sysmon_demand_total = (
                        int(counts.get("DEMAND_TOTAL")) if counts.get("DEMAND_TOTAL") is not None else None
                    )
                except Exception:
                    pass

        # --- Communications (from performance summaries) ---
        comm_sdt = _categorical_counts(perf_main, "communications", "sdt_value")
        comm_hit = int(comm_sdt.get("HIT", 0))
        comm_miss = int(comm_sdt.get("MISS", 0))
        comm_fa = int(comm_sdt.get("FA", 0))
        comm_total = comm_hit + comm_miss + comm_fa
        comm_acc = (comm_hit / comm_total) if comm_total else None

        comm_correct_rate, comm_correct_true, comm_correct_false = _bool_true_rate(
            perf_main, "communications", "correct_radio"
        )
        comm_rt_mean = _numeric_mean(perf_main, "communications", "response_time")

        # --- ResMan tolerance ---
        res_tol_rate, res_tol_true, res_tol_false = _resman_tolerance_rate(perf_main)

        # --- ResMan pump failures (event rows; scheduled difficulty driver) ---
        main_window = _segment_window(summary, main_seg) if main_seg else None
        resman_pump_failures = _count_resman_pump_failures(csv_path, main_window)

        observed_demands = _count_demand_events(csv_path, main_window)

        expected_demands = {"sysmon_failures": None, "comms_radioprompts": None, "resman_pump_failures": None}
        scenario_path = scenario_dir / scenario_filename if scenario_filename else None
        if scenario_path and scenario_path.exists():
            expected_demands = _count_scheduled_demands_in_scenario(scenario_path)

        # --- Subjective TLX (genericscales) ---
        tlx_perf = _get_perf_root(summary, tlx_seg) if tlx_seg else {}
        tlx_mental = _numeric_mean(tlx_perf, "genericscales", "Mental demand")
        tlx_physical = _numeric_mean(tlx_perf, "genericscales", "Physical demand")
        tlx_time = _numeric_mean(tlx_perf, "genericscales", "Time pressure")
        tlx_perf_rating = _numeric_mean(tlx_perf, "genericscales", "Performance")
        tlx_effort = _numeric_mean(tlx_perf, "genericscales", "Effort")
        tlx_frustration = _numeric_mean(tlx_perf, "genericscales", "Frustration")

        tlx_values = [
            v
            for v in [tlx_mental, tlx_physical, tlx_time, tlx_perf_rating, tlx_effort, tlx_frustration]
            if v is not None
        ]
        tlx_mean = (sum(tlx_values) / len(tlx_values)) if tlx_values else None

        rows.append(
            {
                "participant": participant,
                "session": session,
                "seq_id": seq_id,
                "created_at": created_at,
                "mode_verification": bool(mode.get("verification")),
                "mode_pilot1": bool(mode.get("pilot1")),
                "mode_calibration_only": bool(mode.get("calibration_only")),
                "qc_xdf_alignment_status": str(xdf_align.get("status") or ""),
                "block_index": block_index,
                "scenario_name": scenario_name,
                "scenario_filename": scenario_filename,
                "scenario_type": scenario_type,
                "workload_level": level,
                "abort_reason": str(abort_reason or ""),
                "main_segment": main_seg or "",
                "main_duration_sec": _segment_duration_sec(summary, main_seg) if main_seg else None,
                "expected_sysmon_failures": expected_demands.get("sysmon_failures"),
                "expected_comms_radioprompts": expected_demands.get("comms_radioprompts"),
                "expected_resman_pump_failures": expected_demands.get("resman_pump_failures"),
                "observed_sysmon_failures": observed_demands.get("sysmon_failures"),
                "observed_comms_radioprompts": observed_demands.get("comms_radioprompts"),
                "observed_resman_pump_failures": observed_demands.get("resman_pump_failures"),
                "tracking_center_deviation_rmse": tracking_rmse,
                "tracking_n": tracking_n,
                "sysmon_accuracy": sysmon_acc,
                "sysmon_hit": sysmon_hits,
                "sysmon_miss": sysmon_miss,
                "sysmon_fa": sysmon_fa,
                "sysmon_total": sysmon_total,
                "sysmon_demand_total": sysmon_demand_total,
                "comms_accuracy": comm_acc,
                "comms_hit": comm_hit,
                "comms_miss": comm_miss,
                "comms_fa": comm_fa,
                "comms_total": comm_total,
                "comms_correct_radio_rate": comm_correct_rate,
                "comms_correct_radio_true": comm_correct_true,
                "comms_correct_radio_false": comm_correct_false,
                "comms_response_time_mean": comm_rt_mean,
                "resman_tolerance_rate": res_tol_rate,
                "resman_tolerance_true": res_tol_true,
                "resman_tolerance_false": res_tol_false,
                "resman_pump_failures": resman_pump_failures,
                "tlx_segment": tlx_seg or "",
                "tlx_duration_sec": _segment_duration_sec(summary, tlx_seg) if tlx_seg else None,
                "tlx_mental_demand": tlx_mental,
                "tlx_physical_demand": tlx_physical,
                "tlx_time_pressure": tlx_time,
                "tlx_performance": tlx_perf_rating,
                "tlx_effort": tlx_effort,
                "tlx_frustration": tlx_frustration,
                "tlx_mean": tlx_mean,
                "performance_summary_path": str(summary_path),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting a file that might be opened (common on Windows/Excel).
    chosen = out_csv
    if chosen.exists():
        for i in range(1, 1000):
            candidate = chosen.with_name(f"{chosen.stem}_{i}{chosen.suffix}")
            if not candidate.exists():
                chosen = candidate
                break

    with open(chosen, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Best-effort: update a stable "latest" filename for operator convenience.
    latest_path = derived_dir / "pilot_results_summary_latest.csv"
    try:
        with open(chosen, "r", encoding="utf-8", newline="") as src, open(
            latest_path, "w", encoding="utf-8", newline=""
        ) as dst:
            dst.write(src.read())
    except Exception:
        pass

    return chosen


def _find_latest_run_manifest(session_root: Path) -> Path:
    if not session_root.exists():
        raise FileNotFoundError(f"Session root not found: {session_root}")

    candidates = sorted(
        session_root.glob("run_manifest_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No run manifests found under session root. Expected files like: run_manifest_YYYYMMDDTHHMMSS.json"
        )
    return candidates[0]


def _default_output_root() -> Path:
    # Keep aligned with run_openmatb.py default.
    return Path(r"C:\data\adaptive_matb")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a per-scenario pilot performance table as CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-manifest", help="Path to run_manifest_*.json")
    group.add_argument(
        "--session",
        help=(
            "Session ID (e.g., S001). Use with --participant to auto-pick the latest run_manifest_*.json."
        ),
    )
    parser.add_argument(
        "--participant",
        default=None,
        help="Participant ID (e.g., P001). Required when using --session auto mode.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root (default: C:\\data\\adaptive_matb). Used in --session auto mode.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <run_manifest>.performance_table.csv)",
    )
    args = parser.parse_args()

    if args.run_manifest:
        run_manifest_path = Path(args.run_manifest)
        if not run_manifest_path.exists():
            # Common case: launch.json placeholder.
            if "YYYYMMDD" in str(run_manifest_path) or "HHMMSS" in str(run_manifest_path):
                print(
                    "ERROR: run manifest path is still the placeholder timestamp. "
                    "Either edit the run_manifest path, or use auto mode: --participant P001 --session S001",
                )
            else:
                print(f"ERROR: run manifest not found: {run_manifest_path}")

            # Try to help by listing nearby candidates.
            parent = run_manifest_path.parent
            if parent.exists():
                candidates = sorted(parent.glob("run_manifest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    print("Available run manifests in that folder:")
                    for c in candidates[:10]:
                        print(f"  - {c}")
            return 2
    else:
        if not args.participant:
            print("ERROR: --participant is required when using --session auto mode")
            return 2
        output_root = Path(args.output_root) if args.output_root else _default_output_root()
        session_root = output_root / "openmatb" / args.participant / args.session
        try:
            run_manifest_path = _find_latest_run_manifest(session_root)
        except Exception as exc:
            print(f"ERROR: {exc}")
            return 2
        print(f"Using latest run manifest: {run_manifest_path}")

    out_csv = Path(args.out) if args.out else None

    try:
        out_path = export_table(run_manifest_path=run_manifest_path, out_csv=out_csv)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"Wrote performance table CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
