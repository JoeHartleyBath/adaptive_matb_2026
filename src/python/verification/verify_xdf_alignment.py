"""Verify XDF↔CSV marker alignment for Pilot 1 QC.

This script compares OpenMATB markers recorded via LSL (in the .xdf file) against
the CSV log (ground truth from OpenMATB's internal logger) to verify timing accuracy.

Pass criteria:
  - Median absolute marker error ≤ 20 ms
  - 95th percentile absolute error ≤ 50 ms
  - Drift ≤ 5 ms/min

Hard fail:
  - Drift > 20 ms/min
  - Any discontinuities (time jumps, out-of-order timestamps)

Usage:
  python src/python/verification/verify_xdf_alignment.py \\
      --xdf path/to/recording.xdf \\
      --csv path/to/session.csv \\
      --out path/to/qc_report.json

  # Or using run manifest:
  python src/python/verification/verify_xdf_alignment.py \\
      --run-manifest path/to/run_manifest_*.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# pyxdf is required for XDF parsing
try:
    import pyxdf
    HAS_PYXDF = True
except ImportError:
    HAS_PYXDF = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class MarkerEvent:
    """A single marker event with timestamp."""
    name: str
    timestamp: float  # seconds
    source: str  # 'xdf' or 'csv'


@dataclass
class AlignmentResult:
    """Result of marker alignment analysis."""
    n_csv_markers: int
    n_xdf_markers: int
    n_matched: int
    n_unmatched_csv: int
    n_unmatched_xdf: int
    
    # Timing metrics (milliseconds)
    errors_ms: list[float]
    median_abs_error_ms: float
    p95_abs_error_ms: float
    max_abs_error_ms: float
    mean_error_ms: float  # signed, for drift detection
    
    # Drift analysis
    duration_min: float
    drift_ms_per_min: float
    drift_direction: str  # 'xdf_ahead', 'xdf_behind', 'none'
    
    # Discontinuity checks
    has_discontinuities: bool
    discontinuity_details: list[dict]
    
    # Pass/fail
    passed: bool
    fail_reasons: list[str]


def _atomic_write_json(path: Path, payload: dict) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(values: list[float], q: float) -> float:
    """Compute percentile of sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if q <= 0:
        return sorted_vals[0]
    if q >= 100:
        return sorted_vals[-1]
    pos = (q / 100) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_csv_markers(csv_path: Path) -> list[MarkerEvent]:
    """Load LSL markers from OpenMATB session CSV."""
    markers = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_type = (row.get("type") or "").strip().lower()
            module = (row.get("module") or "").strip().lower()
            address = (row.get("address") or "").strip().lower()
            
            # LSL markers are logged as: type=event, module=labstreaminglayer, address=marker
            if row_type != "event" or module != "labstreaminglayer" or address != "marker":
                continue
            
            # Use scenario_time as the timestamp (OpenMATB internal clock)
            time_str = row.get("scenario_time", "")
            try:
                timestamp = float(time_str)
            except (ValueError, TypeError):
                continue
            
            name = (row.get("value") or "").strip()
            if not name:
                continue
            
            # Strip payload after '|' for matching
            name_clean = name.split("|", 1)[0].strip()
            markers.append(MarkerEvent(name=name_clean, timestamp=timestamp, source="csv"))
    
    return markers


def load_xdf_markers(xdf_path: Path, stream_name: str = "OpenMATB", stream_type: str = "Markers") -> list[MarkerEvent]:
    """Load markers from XDF file."""
    if not HAS_PYXDF:
        raise ImportError("pyxdf is required for XDF parsing. Install with: pip install pyxdf")
    
    streams, header = pyxdf.load_xdf(str(xdf_path))
    
    markers = []
    marker_stream = None
    
    for stream in streams:
        info = stream["info"]
        name = info.get("name", [""])[0] if isinstance(info.get("name"), list) else info.get("name", "")
        stype = info.get("type", [""])[0] if isinstance(info.get("type"), list) else info.get("type", "")
        
        if name == stream_name or stype == stream_type:
            marker_stream = stream
            break
    
    if marker_stream is None:
        print(f"WARNING: No marker stream found (name={stream_name}, type={stream_type})", file=sys.stderr)
        return markers
    
    timestamps = marker_stream.get("time_stamps", [])
    time_series = marker_stream.get("time_series", [])
    
    for i, ts in enumerate(timestamps):
        if i < len(time_series):
            # Marker value is typically in first channel
            val = time_series[i]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            name = str(val).strip()
            # Strip payload after '|'
            name_clean = name.split("|", 1)[0].strip()
            markers.append(MarkerEvent(name=name_clean, timestamp=float(ts), source="xdf"))
    
    return markers


def check_discontinuities(markers: list[MarkerEvent], max_gap_s: float = 60.0) -> list[dict]:
    """Check for timestamp discontinuities (jumps or out-of-order)."""
    if len(markers) < 2:
        return []
    
    issues = []
    sorted_markers = sorted(markers, key=lambda m: m.timestamp)
    
    for i in range(1, len(sorted_markers)):
        prev = sorted_markers[i - 1]
        curr = sorted_markers[i]
        
        delta = curr.timestamp - prev.timestamp
        
        # Out-of-order (should not happen after sorting, but check original order)
        if delta < 0:
            issues.append({
                "type": "out_of_order",
                "index": i,
                "prev_name": prev.name,
                "curr_name": curr.name,
                "delta_s": delta,
            })
        
        # Large gap (potential recording interruption)
        if delta > max_gap_s:
            issues.append({
                "type": "large_gap",
                "index": i,
                "prev_name": prev.name,
                "prev_time": prev.timestamp,
                "curr_name": curr.name,
                "curr_time": curr.timestamp,
                "gap_s": delta,
            })
    
    # Check original ordering
    for i in range(1, len(markers)):
        if markers[i].timestamp < markers[i - 1].timestamp:
            issues.append({
                "type": "original_order_violation",
                "index": i,
                "message": f"Marker {i} timestamp < marker {i-1} timestamp",
            })
    
    return issues


def match_markers(csv_markers: list[MarkerEvent], xdf_markers: list[MarkerEvent], 
                  max_offset_s: float = 2.0) -> list[tuple[MarkerEvent, MarkerEvent, float]]:
    """Match CSV and XDF markers by name and approximate time.
    
    Returns list of (csv_marker, xdf_marker, error_ms) tuples.
    """
    matches = []
    used_xdf_indices = set()
    
    for csv_m in csv_markers:
        best_match = None
        best_error = float('inf')
        best_idx = -1
        
        for j, xdf_m in enumerate(xdf_markers):
            if j in used_xdf_indices:
                continue
            if csv_m.name != xdf_m.name:
                continue
            
            # XDF timestamps are absolute LSL time; CSV timestamps are scenario time.
            # We need to find the best match within a reasonable offset window.
            # For now, match by order within same-name markers.
            error = abs(csv_m.timestamp - xdf_m.timestamp)
            if error < best_error and error < max_offset_s * 1000:  # Allow large initial offset
                best_error = error
                best_match = xdf_m
                best_idx = j
        
        if best_match is not None:
            used_xdf_indices.add(best_idx)
            # Error in milliseconds
            error_ms = (best_match.timestamp - csv_m.timestamp) * 1000
            matches.append((csv_m, best_match, error_ms))
    
    return matches


def compute_drift(matches: list[tuple[MarkerEvent, MarkerEvent, float]]) -> tuple[float, float, str]:
    """Compute drift from matched markers.
    
    Returns (drift_ms_per_min, duration_min, direction).
    """
    if len(matches) < 2:
        return 0.0, 0.0, "none"
    
    # Sort by CSV timestamp
    sorted_matches = sorted(matches, key=lambda m: m[0].timestamp)
    
    first = sorted_matches[0]
    last = sorted_matches[-1]
    
    duration_s = last[0].timestamp - first[0].timestamp
    if duration_s < 10:  # Need at least 10 seconds for meaningful drift
        return 0.0, duration_s / 60.0, "none"
    
    duration_min = duration_s / 60.0
    
    # Drift = change in error over time
    first_error_ms = first[2]
    last_error_ms = last[2]
    total_drift_ms = last_error_ms - first_error_ms
    drift_ms_per_min = total_drift_ms / duration_min
    
    if abs(drift_ms_per_min) < 0.1:
        direction = "none"
    elif drift_ms_per_min > 0:
        direction = "xdf_ahead"
    else:
        direction = "xdf_behind"
    
    return drift_ms_per_min, duration_min, direction


def analyze_alignment(
    csv_markers: list[MarkerEvent],
    xdf_markers: list[MarkerEvent],
    thresholds: Optional[dict] = None,
) -> AlignmentResult:
    """Perform full alignment analysis."""
    
    if thresholds is None:
        thresholds = {
            "median_abs_ms": 20,
            "p95_abs_ms": 50,
            "drift_ms_per_min": 5,
            "hard_fail_drift_ms_per_min": 20,
        }
    
    # Check discontinuities in XDF markers
    discontinuities = check_discontinuities(xdf_markers)
    
    # Match markers
    matches = match_markers(csv_markers, xdf_markers)
    
    # Extract errors
    errors_ms = [m[2] for m in matches]
    abs_errors_ms = [abs(e) for e in errors_ms]
    
    # Compute statistics
    if abs_errors_ms:
        median_abs = _percentile(abs_errors_ms, 50)
        p95_abs = _percentile(abs_errors_ms, 95)
        max_abs = max(abs_errors_ms)
        mean_error = sum(errors_ms) / len(errors_ms)
    else:
        median_abs = 0.0
        p95_abs = 0.0
        max_abs = 0.0
        mean_error = 0.0
    
    # Compute drift
    drift_ms_per_min, duration_min, drift_direction = compute_drift(matches)
    
    # Determine pass/fail
    fail_reasons = []
    
    if discontinuities:
        fail_reasons.append(f"Found {len(discontinuities)} discontinuities in XDF marker stream")
    
    if median_abs > thresholds["median_abs_ms"]:
        fail_reasons.append(f"Median absolute error {median_abs:.1f} ms > {thresholds['median_abs_ms']} ms threshold")
    
    if p95_abs > thresholds["p95_abs_ms"]:
        fail_reasons.append(f"95th percentile error {p95_abs:.1f} ms > {thresholds['p95_abs_ms']} ms threshold")
    
    if abs(drift_ms_per_min) > thresholds["hard_fail_drift_ms_per_min"]:
        fail_reasons.append(f"Drift {abs(drift_ms_per_min):.2f} ms/min > {thresholds['hard_fail_drift_ms_per_min']} ms/min HARD FAIL threshold")
    elif abs(drift_ms_per_min) > thresholds["drift_ms_per_min"]:
        fail_reasons.append(f"Drift {abs(drift_ms_per_min):.2f} ms/min > {thresholds['drift_ms_per_min']} ms/min threshold")
    
    if len(matches) == 0 and len(csv_markers) > 0:
        fail_reasons.append("No markers matched between CSV and XDF")
    
    passed = len(fail_reasons) == 0
    
    return AlignmentResult(
        n_csv_markers=len(csv_markers),
        n_xdf_markers=len(xdf_markers),
        n_matched=len(matches),
        n_unmatched_csv=len(csv_markers) - len(matches),
        n_unmatched_xdf=len(xdf_markers) - len(matches),
        errors_ms=errors_ms,
        median_abs_error_ms=median_abs,
        p95_abs_error_ms=p95_abs,
        max_abs_error_ms=max_abs,
        mean_error_ms=mean_error,
        duration_min=duration_min,
        drift_ms_per_min=drift_ms_per_min,
        drift_direction=drift_direction,
        has_discontinuities=len(discontinuities) > 0,
        discontinuity_details=discontinuities,
        passed=passed,
        fail_reasons=fail_reasons,
    )


def run_qc(
    xdf_path: Path,
    csv_paths: list[Path],
    thresholds: Optional[dict] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run full QC and optionally write report."""
    
    if not HAS_PYXDF:
        return {
            "status": "error",
            "error": "pyxdf not installed. Run: pip install pyxdf",
        }
    
    # Load XDF markers
    try:
        xdf_markers = load_xdf_markers(xdf_path)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to load XDF: {e}",
        }
    
    # Load and combine CSV markers from all scenario CSVs
    all_csv_markers = []
    csv_load_errors = []
    for csv_path in csv_paths:
        try:
            markers = load_csv_markers(csv_path)
            all_csv_markers.extend(markers)
        except Exception as e:
            csv_load_errors.append({"path": str(csv_path), "error": str(e)})
    
    # Analyze alignment
    result = analyze_alignment(all_csv_markers, xdf_markers, thresholds)
    
    # Build report
    report = {
        "schema": "xdf_alignment_qc_v0",
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "xdf_path": str(xdf_path),
            "csv_paths": [str(p) for p in csv_paths],
            "csv_load_errors": csv_load_errors,
        },
        "thresholds": thresholds or {
            "median_abs_ms": 20,
            "p95_abs_ms": 50,
            "drift_ms_per_min": 5,
            "hard_fail_drift_ms_per_min": 20,
        },
        "results": {
            "passed": result.passed,
            "fail_reasons": result.fail_reasons,
            "n_csv_markers": result.n_csv_markers,
            "n_xdf_markers": result.n_xdf_markers,
            "n_matched": result.n_matched,
            "n_unmatched_csv": result.n_unmatched_csv,
            "n_unmatched_xdf": result.n_unmatched_xdf,
            "timing": {
                "median_abs_error_ms": result.median_abs_error_ms,
                "p95_abs_error_ms": result.p95_abs_error_ms,
                "max_abs_error_ms": result.max_abs_error_ms,
                "mean_error_ms": result.mean_error_ms,
            },
            "drift": {
                "duration_min": result.duration_min,
                "drift_ms_per_min": result.drift_ms_per_min,
                "direction": result.drift_direction,
            },
            "discontinuities": {
                "found": result.has_discontinuities,
                "count": len(result.discontinuity_details),
                "details": result.discontinuity_details[:10],  # Limit to first 10
            },
        },
        "status": "passed" if result.passed else "failed",
    }
    
    if output_path:
        _atomic_write_json(output_path, report)
        print(f"Wrote QC report: {output_path}")
    
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify XDF↔CSV marker alignment for Pilot 1 QC.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-manifest", help="Path to run manifest JSON (contains XDF and CSV paths)")
    group.add_argument("--xdf", help="Path to XDF file")
    
    parser.add_argument("--csv", nargs="+", help="Path(s) to CSV file(s) (required if not using --run-manifest)")
    parser.add_argument("--out", help="Output path for QC report JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except errors")
    
    args = parser.parse_args()
    
    if not HAS_PYXDF:
        print("ERROR: pyxdf is required. Install with: pip install pyxdf", file=sys.stderr)
        return 2
    
    # Determine inputs
    if args.run_manifest:
        manifest = _load_json(Path(args.run_manifest))
        xdf_path = Path(manifest.get("physiology", {}).get("xdf_path", ""))
        if not xdf_path or not xdf_path.exists():
            print(f"ERROR: XDF path not found in manifest or file missing: {xdf_path}", file=sys.stderr)
            return 2
        
        csv_paths = []
        for scenario in manifest.get("playlist", []):
            csv_str = scenario.get("session_csv", "")
            if csv_str:
                csv_paths.append(Path(csv_str))
        
        thresholds = manifest.get("qc", {}).get("xdf_alignment", {}).get("thresholds")
        out_path = Path(args.out) if args.out else Path(args.run_manifest).with_suffix(".qc_alignment.json")
    else:
        if not args.csv:
            print("ERROR: --csv is required when not using --run-manifest", file=sys.stderr)
            return 2
        xdf_path = Path(args.xdf)
        csv_paths = [Path(p) for p in args.csv]
        thresholds = None
        out_path = Path(args.out) if args.out else None
    
    # Validate paths
    if not xdf_path.exists():
        print(f"ERROR: XDF file not found: {xdf_path}", file=sys.stderr)
        return 2
    
    missing_csvs = [p for p in csv_paths if not p.exists()]
    if missing_csvs:
        print(f"WARNING: Some CSV files not found: {missing_csvs}", file=sys.stderr)
        csv_paths = [p for p in csv_paths if p.exists()]
    
    if not csv_paths:
        print("ERROR: No valid CSV files to compare", file=sys.stderr)
        return 2
    
    # Run QC
    report = run_qc(xdf_path, csv_paths, thresholds, out_path)
    
    if report.get("status") == "error":
        print(f"ERROR: {report.get('error')}", file=sys.stderr)
        return 2
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"XDF↔CSV Marker Alignment QC")
        print(f"{'='*60}")
        print(f"XDF: {xdf_path}")
        print(f"CSVs: {len(csv_paths)} file(s)")
        print(f"\nResults:")
        results = report.get("results", {})
        print(f"  Markers matched: {results.get('n_matched', 0)} / {results.get('n_csv_markers', 0)} CSV")
        timing = results.get("timing", {})
        print(f"  Median abs error: {timing.get('median_abs_error_ms', 0):.1f} ms")
        print(f"  95th percentile:  {timing.get('p95_abs_error_ms', 0):.1f} ms")
        drift = results.get("drift", {})
        print(f"  Drift: {drift.get('drift_ms_per_min', 0):.2f} ms/min over {drift.get('duration_min', 0):.1f} min")
        discontinuities = results.get("discontinuities", {})
        print(f"  Discontinuities: {discontinuities.get('count', 0)}")
        print(f"\nStatus: {'PASSED ✓' if results.get('passed') else 'FAILED ✗'}")
        if not results.get("passed"):
            print("Fail reasons:")
            for reason in results.get("fail_reasons", []):
                print(f"  - {reason}")
        print(f"{'='*60}\n")
    
    return 0 if report.get("results", {}).get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
