"""summarise OpenMATB performance rows from the session CSV.

OpenMATB writes a single event log CSV per run (see manifest key: paths.session_csv).
Task "performance" is tracked as rows where:
  type == 'performance'
  module == plugin name (e.g., sysmon/track/communications/resman)
  address == metric name
  value == metric value (numeric/bool/categorical)

This script derives a compact JSON summary (overall + per marker-defined segment).

Usage:
  python src/python/summarise_openmatb_performance.py --manifest <path/to/*.manifest.json>
  python src/python/summarise_openmatb_performance.py --csv <path/to/*.csv> --out <summary.json>

Notes:
- Segmenting uses labstreaminglayer markers of the form:
    STUDY/.../START|pid=...  and  STUDY/.../END|pid=...
  Only the portion before the first '|' is used for segment names.
- Designed to run in a pure-stdlib Python environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Marker:
    name: str
    time_sec: float


@dataclass(frozen=True)
class Segment:
    name: str
    start_sec: float
    end_sec: float


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
    tmp.replace(path)


def _parse_bool(value: str) -> Optional[bool]:
    v = value.strip().lower()
    if v in {"true", "1", "yes"}:
        return True
    if v in {"false", "0", "no"}:
        return False
    return None


def _parse_float(value: str) -> Optional[float]:
    v = value.strip()
    if not v:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    return f


def _is_nan(x: float) -> bool:
    try:
        return math.isnan(x)
    except Exception:
        return False


def _percentile(sorted_values: list[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])
    # Linear interpolation between closest ranks
    pos = (q / 100) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def _metric_summary(values: list[str]) -> dict[str, Any]:
    """summarise a list of raw string values from CSV."""

    bool_values: list[bool] = []
    numeric_values: list[float] = []
    nan_count = 0
    categorical: dict[str, int] = {}

    for raw in values:
        raw_str = "" if raw is None else str(raw)
        b = _parse_bool(raw_str)
        if b is not None:
            bool_values.append(b)
            continue

        f = _parse_float(raw_str)
        if f is not None:
            if _is_nan(f):
                nan_count += 1
            else:
                numeric_values.append(float(f))
            continue

        key = raw_str.strip()
        categorical[key] = categorical.get(key, 0) + 1

    out: dict[str, Any] = {
        "count": len(values),
        "nan_count": nan_count,
    }

    if bool_values:
        out["boolean"] = {
            "true": sum(1 for v in bool_values if v),
            "false": sum(1 for v in bool_values if not v),
        }

    if numeric_values:
        numeric_values.sort()
        out["numeric"] = {
            "count": len(numeric_values),
            "min": numeric_values[0],
            "max": numeric_values[-1],
            "mean": sum(numeric_values) / len(numeric_values),
            "median": _percentile(numeric_values, 50),
            "p10": _percentile(numeric_values, 10),
            "p90": _percentile(numeric_values, 90),
        }

    if categorical and not bool_values and not numeric_values:
        # Keep stable ordering by count desc then key asc.
        out["categorical"] = dict(
            sorted(categorical.items(), key=lambda kv: (-kv[1], kv[0]))
        )

    if categorical and (bool_values or numeric_values):
        # Mixed types: still useful to see string categories.
        out["categorical"] = dict(
            sorted(categorical.items(), key=lambda kv: (-kv[1], kv[0]))
        )

    return out


def _parse_float_list(values: list[str]) -> list[float]:
    out: list[float] = []
    for raw in values:
        f = _parse_float("" if raw is None else str(raw))
        if f is None:
            continue
        if _is_nan(f):
            continue
        out.append(float(f))
    return out


def _rmse(values: list[float]) -> Optional[float]:
    if not values:
        return None
    mean_sq = sum(v * v for v in values) / len(values)
    return math.sqrt(mean_sq)


def _compute_derived_kpis(by_module: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    """Compute a small set of derived KPIs from performance rows.

    These are intentionally conservative and transparent, so KPI definitions can
    evolve without changing the raw logging.
    """

    derived: dict[str, Any] = {}

    # Tracking RMSE
    track = by_module.get("track")
    if isinstance(track, dict):
        deviations_raw = track.get("center_deviation")
        if isinstance(deviations_raw, list):
            deviations = _parse_float_list(deviations_raw)
            derived["tracking"] = {
                "center_deviation_rmse": _rmse(deviations),
                "n": len(deviations),
            }

    # SysMon response accuracy
    sysmon = by_module.get("sysmon")
    if isinstance(sysmon, dict):
        sdt_raw = sysmon.get("signal_detection")
        if isinstance(sdt_raw, list):
            counts = {"HIT": 0, "MISS": 0, "FA": 0}
            other = 0
            for v in sdt_raw:
                key = ("" if v is None else str(v)).strip().upper()
                if key in counts:
                    counts[key] += 1
                else:
                    other += 1

            total = sum(counts.values())
            accuracy = (counts["HIT"] / total) if total else None
            derived["sysmon"] = {
                "accuracy": accuracy,
                "counts": {
                    **counts,
                    "OTHER": other,
                    "TOTAL": total,
                    "DEMAND_TOTAL": counts["HIT"] + counts["MISS"],
                },
            }

    # Communications SDT + common failure mode warning
    comms = by_module.get("communications")
    if isinstance(comms, dict):
        sdt_raw = comms.get("sdt_value")
        responded_radio_raw = comms.get("responded_radio")
        responded_frequency_raw = comms.get("responded_frequency")
        response_time_raw = comms.get("response_time")

        if isinstance(sdt_raw, list):
            counts = {"HIT": 0, "MISS": 0, "FA": 0}
            other = 0
            for v in sdt_raw:
                key = ("" if v is None else str(v)).strip().upper()
                if key in counts:
                    counts[key] += 1
                else:
                    other += 1

            total = sum(counts.values())
            accuracy = (counts["HIT"] / total) if total else None
            comms_out: dict[str, Any] = {
                "accuracy": accuracy,
                "counts": {
                    **counts,
                    "OTHER": other,
                    "TOTAL": total,
                    "DEMAND_TOTAL": counts["HIT"] + counts["MISS"],
                },
            }

            responded_radio = (
                _parse_float_list(responded_radio_raw)
                if isinstance(responded_radio_raw, list)
                else []
            )
            responded_frequency = (
                _parse_float_list(responded_frequency_raw)
                if isinstance(responded_frequency_raw, list)
                else []
            )
            response_times = (
                _parse_float_list(response_time_raw)
                if isinstance(response_time_raw, list)
                else []
            )

            if (
                total > 0
                and counts["MISS"] == total
                and not responded_radio
                and not responded_frequency
                and not response_times
            ):
                comms_out["warning"] = (
                    "All communications trials are MISS with NaN responded_* and response_time. "
                    "This often means responses were not validated/submitted. "
                    "Default validate key is ENTER (press after selecting radio + tuning frequency)."
                )

            derived["communications"] = comms_out

    return derived


def _read_markers(csv_path: Path) -> list[Marker]:
    markers: list[Marker] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            row_type = (row.get("type") or "").strip().lower()
            module = (row.get("module") or "").strip().lower()
            address = (row.get("address") or "").strip().lower()
            if row_type != "event" or module != "labstreaminglayer" or address != "marker":
                continue

            t = _parse_float(str(row.get("scenario_time") or ""))
            if t is None:
                continue
            raw = str(row.get("value") or "")
            name = raw.split("|", 1)[0].strip()
            if not name:
                continue
            markers.append(Marker(name=name, time_sec=float(t)))

    # Keep the first occurrence per marker name (START/END can repeat, but we want deterministic segments).
    # If you later want multiple repetitions, we can extend this to pair in sequence.
    first: dict[str, Marker] = {}
    for m in markers:
        if m.name not in first:
            first[m.name] = m
    return sorted(first.values(), key=lambda m: m.time_sec)


def _derive_segments_from_markers(markers: list[Marker]) -> list[Segment]:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}

    for m in markers:
        if m.name.endswith("/START"):
            base = m.name[: -len("/START")]
            starts[base] = m.time_sec
        elif m.name.endswith("/END"):
            base = m.name[: -len("/END")]
            ends[base] = m.time_sec

    segments: list[Segment] = []
    for base, start in starts.items():
        end = ends.get(base)
        if end is None:
            continue
        if end <= start:
            continue
        segments.append(Segment(name=base, start_sec=start, end_sec=end))

    return sorted(segments, key=lambda s: s.start_sec)


def _collect_performance_rows(
    csv_path: Path,
    *,
    window: Optional[tuple[float, float]] = None,
) -> dict[str, dict[str, list[str]]]:
    """Return module -> metric -> list of raw values."""

    by_module: dict[str, dict[str, list[str]]] = {}

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=2):
            row_type = (row.get("type") or "").strip().lower()
            if row_type != "performance":
                continue

            t = _parse_float(str(row.get("scenario_time") or ""))
            if t is None:
                continue
            if window is not None:
                start, end = window
                if not (start <= float(t) <= end):
                    continue

            module = (row.get("module") or "").strip().lower() or "(unknown)"
            metric = (row.get("address") or "").strip() or "(unknown)"
            value = "" if row.get("value") is None else str(row.get("value"))

            metrics = by_module.setdefault(module, {})
            metrics.setdefault(metric, []).append(value)

    return by_module


def _summarise_performance(by_module: dict[str, dict[str, list[str]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for module, metrics in sorted(by_module.items()):
        mod_summary: dict[str, Any] = {}
        for metric, values in sorted(metrics.items()):
            mod_summary[metric] = _metric_summary(values)
        summary[module] = mod_summary
    return summary


def summarise_csv(csv_path: Path) -> dict[str, Any]:
    markers = _read_markers(csv_path)
    segments = _derive_segments_from_markers(markers)

    overall_rows = _collect_performance_rows(csv_path)
    overall = _summarise_performance(overall_rows)
    overall_kpis = _compute_derived_kpis(overall_rows)

    per_segment: dict[str, Any] = {}
    for seg in segments:
        seg_rows = _collect_performance_rows(csv_path, window=(seg.start_sec, seg.end_sec))
        per_segment[seg.name] = {
            "window": {"start_sec": seg.start_sec, "end_sec": seg.end_sec},
            "performance": _summarise_performance(seg_rows),
            "derived_kpis": _compute_derived_kpis(seg_rows),
        }

    return {
        "schema_version": 2,
        "csv_path": str(csv_path),
        "markers": [{"name": m.name, "time_sec": m.time_sec} for m in markers],
        "segments": [{"name": s.name, "start_sec": s.start_sec, "end_sec": s.end_sec} for s in segments],
        "overall": overall,
        "derived_kpis": overall_kpis,
        "per_segment": per_segment,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="summarise OpenMATB performance metrics from a session CSV.")
    parser.add_argument("--manifest", default=None, help="Path to *.manifest.json (preferred input).")
    parser.add_argument("--csv", default=None, help="Path to the session CSV (alternative input).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: next to manifest/CSV as *.performance_summary.json).",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else None
    csv_path = Path(args.csv) if args.csv else None

    if manifest_path is None and csv_path is None:
        print("ERROR: Provide --manifest or --csv")
        return 2

    manifest: Optional[dict[str, Any]] = None
    if manifest_path is not None:
        if not manifest_path.exists():
            print(f"ERROR: manifest not found: {manifest_path}")
            return 2
        manifest = _load_json(manifest_path)
        csv_path_str = (
            (manifest.get("paths") or {}).get("session_csv")
            if isinstance(manifest.get("paths"), dict)
            else None
        )
        if csv_path_str:
            csv_path = Path(str(csv_path_str))

    if csv_path is None:
        print("ERROR: Could not determine CSV path")
        return 2

    if not csv_path.exists():
        print(f"ERROR: session CSV not found: {csv_path}")
        return 2

    summary = summarise_csv(csv_path)

    # Attach a minimal run context when available.
    if manifest is not None:
        summary["manifest_path"] = str(manifest_path)
        summary["identifiers"] = manifest.get("identifiers")
        summary["scenario_name"] = manifest.get("scenario_name")
        summary["repo_commit"] = manifest.get("repo_commit")
        summary["submodule_commit"] = manifest.get("submodule_commit")

    if args.out:
        out_path = Path(args.out)
    else:
        if manifest_path is not None:
            out_path = manifest_path.with_suffix(manifest_path.suffix + ".performance_summary.json")
        else:
            out_path = csv_path.with_suffix(csv_path.suffix + ".performance_summary.json")

    _atomic_write_json(out_path, summary)
    print(f"Wrote performance summary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
