"""Post-session EEG audit — run after a complete study session.

Checks all recordings in a physio directory for signal quality and (optionally)
cross-session feature consistency.  Writes a JSON summary for later aggregation
across participants.

Checks
------
  Per recording   : duration, channel count, flat/noisy channels, timestamp gaps
  With --model-dir: model artefact completeness, per-recording feature distribution
                    (normalised), C1-vs-C2 consistency, adaptation/control drift

Usage
-----
    python scripts/analysis/audit_session_eeg.py \\
        --physio-dir "C:/data/adaptive_matb/physiology/sub-P001/ses-S001/physio" \\
        [--model-dir "C:/data/adaptive_matb/models/P001"] \\
        [--pid P001]

Output
------
    Printed summary tables
    results/qc/{pid}_{session}_session_audit.json   (always written)

Exit codes
----------
    0  all recordings OK
    1  one or more recordings have flagged channels
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyxdf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg import EegPreprocessor, extract_windows          # noqa: E402
from eeg.extract_features import _build_region_map, _extract_feat  # noqa: E402
from eeg.xdf_loader import (                               # noqa: E402
    PREPROCESSING_CONFIG,
    WINDOW_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_FLAT_UV       = 0.5
_NOISY_UV      = 200.0
_STREAM_TO_UV  = 1e6     # ANT eego streams in volts via LSL
_TIMESTAMP_TOL = 0.10
_DEFAULT_REGION_CFG = _REPO_ROOT / "config" / "eeg_feature_extraction.yaml"

# Known acquisition types, in session order.  Canonical XDFs match
# *_acq-{acq}_physio.xdf  (the glob excludes _old1 files automatically).
_ACQUISITIONS = ["rest", "cal_c1", "cal_c2", "adaptation", "control"]

# Only task XDFs are used for feature checks (rest has no task windows).
_FEATURE_ACQUISITIONS = ["cal_c1", "cal_c2", "adaptation", "control"]

_MODEL_ARTEFACTS = ["pipeline.pkl", "selector.pkl", "norm_stats.json"]
_EXPECTED_NORM_LEN = 54  # features selected by SelectKBest k=35 → 40 after last calibration

_QC_DIR = _REPO_ROOT / "results" / "qc"


# ---------------------------------------------------------------------------
# Per-XDF helpers
# ---------------------------------------------------------------------------

def _load_and_decimate(xdf_path: Path) -> tuple[np.ndarray, np.ndarray, float, bool, int] | None:
    """Load XDF, merge EEG streams, decimate to target srate.

    Returns (raw_ts, eeg_ts, actual_srate, decimated, factor) or None on error.
    """
    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"    ERROR loading {xdf_path.name}: {exc}")
        return None

    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        print(f"    ERROR: no EEG stream in {xdf_path.name}")
        return None

    raw_ts = np.array(eeg_stream["time_series"], dtype=np.float64).T
    eeg_ts = np.array(eeg_stream["time_stamps"])

    actual_srate = (
        (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
        if len(eeg_ts) > 1
        else float(eeg_stream["info"]["nominal_srate"][0])
    )
    target_srate = PREPROCESSING_CONFIG.srate
    decimated = False
    factor = 1
    if actual_srate > target_srate * 1.1:
        factor = int(round(actual_srate / target_srate))
        raw_ts = raw_ts[:, ::factor]
        eeg_ts = eeg_ts[::factor]
        decimated = True

    return raw_ts, eeg_ts, actual_srate, decimated, factor


def _check_timestamps(ts: np.ndarray, target_srate: float) -> tuple[bool, int, float]:
    """Return (ok, n_gaps, max_gap_s)."""
    if len(ts) < 2:
        return True, 0, 0.0
    diffs = np.diff(ts)
    expected = 1.0 / target_srate
    n_gaps = int(np.sum(diffs > 2.0 * expected))
    max_gap_s = float(diffs.max())
    median_interval = float(np.median(diffs))
    deviation = abs(median_interval - expected) / expected
    ok = deviation <= _TIMESTAMP_TOL and n_gaps == 0
    return ok, n_gaps, max_gap_s


def _audit_xdf(xdf_path: Path, ch_names: list[str]) -> dict:
    """Run channel and timestamp checks on one XDF.  Returns a result dict."""
    result: dict = {"found": True, "path": str(xdf_path)}

    loaded = _load_and_decimate(xdf_path)
    if loaded is None:
        result["load_error"] = True
        result["ok"] = False
        return result

    raw_ts, eeg_ts, actual_srate, decimated, factor = loaded
    n_ch, n_samples = raw_ts.shape
    target_srate = PREPROCESSING_CONFIG.srate
    duration_s = n_samples / target_srate

    result["duration_s"] = round(duration_s, 1)
    result["n_channels"] = n_ch
    result["srate_actual"] = round(actual_srate, 1)
    result["decimated"] = decimated

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(n_ch, prewarm=raw_ts[:, 0])
    prep_ts = preprocessor.process(raw_ts)

    prep_rms = np.sqrt(np.mean(prep_ts ** 2, axis=1)) * _STREAM_TO_UV
    prep_std = prep_ts.std(axis=1) * _STREAM_TO_UV

    flat_idx  = [i for i in range(n_ch) if prep_std[i]  < _FLAT_UV]
    noisy_idx = [i for i in range(n_ch) if prep_rms[i] > _NOISY_UV]

    result["flat_channels"]   = [ch_names[i] for i in flat_idx]
    result["noisy_channels"]  = [ch_names[i] for i in noisy_idx]
    result["median_prep_rms_uv"] = round(float(np.median(prep_rms)), 1)

    ts_ok, n_gaps, max_gap_s = _check_timestamps(eeg_ts, target_srate)
    result["timestamp_ok"]     = ts_ok
    result["timestamp_n_gaps"] = n_gaps
    result["timestamp_max_gap_s"] = round(max_gap_s, 4)

    n_bad = len(flat_idx) + len(noisy_idx)
    result["ok"] = n_bad == 0 and ts_ok
    return result


def _extract_features_zscored(
    xdf_path: Path,
    ch_names: list[str],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> np.ndarray | None:
    """Load, preprocess, window, extract features, Z-normalise.

    Returns X_z of shape (N_windows, N_features) or None on error.
    """
    loaded = _load_and_decimate(xdf_path)
    if loaded is None:
        return None
    raw_ts, _, _, _, _ = loaded

    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(raw_ts.shape[0])
    prep_ts = preprocessor.process(raw_ts)

    windows = extract_windows(prep_ts, WINDOW_CONFIG)
    if len(windows) == 0:
        return None

    region_map = _build_region_map(_DEFAULT_REGION_CFG, ch_names)
    X, _ = _extract_feat(windows, PREPROCESSING_CONFIG.srate, region_map)
    return (X - norm_mean) / norm_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-session EEG audit.  Checks all recordings in a physio directory "
            "and writes a JSON summary."
        )
    )
    parser.add_argument(
        "--physio-dir", required=True, type=Path,
        help="Physio directory containing session XDFs (e.g. sub-P001/ses-S001/physio/).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Calibrated model directory (enables feature-level cross-session checks).",
    )
    parser.add_argument(
        "--pid", type=str, default=None,
        help="Participant ID (inferred from directory name if omitted).",
    )
    args = parser.parse_args()

    if not args.physio_dir.exists():
        sys.exit(f"ERROR: physio-dir not found: {args.physio_dir}")

    # ------------------------------------------------------------------
    # Infer PID and session from directory structure
    # ------------------------------------------------------------------
    # Expected: .../sub-{PID}/ses-{SES}/physio
    try:
        session = args.physio_dir.parent.name.removeprefix("ses-")
        pid_from_path = args.physio_dir.parent.parent.name.removeprefix("sub-")
    except Exception:
        session = "unknown"
        pid_from_path = "unknown"

    pid = args.pid or pid_from_path

    print(f"\nSession EEG audit — PID: {pid}  Session: {session}")
    print(f"Physio dir : {args.physio_dir}")
    print(f"Model dir  : {args.model_dir or '(not provided)'}\n")

    ch_names = _load_eeg_metadata(_REPO_ROOT)

    # ------------------------------------------------------------------
    # Discover XDFs
    # ------------------------------------------------------------------
    xdf_paths: dict[str, Path | None] = {}
    for acq in _ACQUISITIONS:
        matches = sorted(args.physio_dir.glob(f"*_acq-{acq}_physio.xdf"))
        xdf_paths[acq] = matches[0] if matches else None

    # ------------------------------------------------------------------
    # Per-XDF channel + timestamp checks
    # ------------------------------------------------------------------
    print("=" * 72)
    print("RECORDING QUALITY")
    print("=" * 72)
    col = f"{'Recording':<12}  {'Duration':>9}  {'Flat':>5}  {'Noisy':>6}  {'Med RMS':>8}  {'Timestamps':<18}  Status"
    print(col)
    print("-" * 72)

    recording_results: dict[str, dict] = {}
    any_bad = False

    for acq in _ACQUISITIONS:
        path = xdf_paths[acq]
        if path is None:
            recording_results[acq] = {"found": False, "ok": False}
            print(f"{acq:<12}  {'NOT FOUND':>9}")
            any_bad = True
            continue

        print(f"{acq:<12}  checking ...", end="\r", flush=True)
        res = _audit_xdf(path, ch_names)
        recording_results[acq] = res

        if res.get("load_error"):
            print(f"{acq:<12}  LOAD ERROR")
            any_bad = True
            continue

        dur_str = f"{res['duration_s']:.1f} s"
        n_flat   = len(res["flat_channels"])
        n_noisy  = len(res["noisy_channels"])
        rms_str  = f"{res['median_prep_rms_uv']:.1f} µV"
        ts_str   = "OK" if res["timestamp_ok"] else f"{res['timestamp_n_gaps']} gap(s)"
        status   = "OK" if res["ok"] else "WARN"

        print(
            f"{acq:<12}  {dur_str:>9}  {n_flat:>5}  {n_noisy:>6}  "
            f"{rms_str:>8}  {ts_str:<18}  {status}"
        )
        if not res["ok"]:
            any_bad = True
            if n_flat:
                print(f"  Flat    : {', '.join(res['flat_channels'])}")
            if n_noisy:
                print(f"  Noisy   : {', '.join(res['noisy_channels'])}")
            if not res["timestamp_ok"]:
                print(
                    f"  Timestamps: {res['timestamp_n_gaps']} gap(s), "
                    f"max {res['timestamp_max_gap_s']:.3f} s"
                )

    # ------------------------------------------------------------------
    # Model artefact check + feature-level cross-session checks
    # ------------------------------------------------------------------
    model_result: dict = {}

    if args.model_dir is not None:
        print(f"\n{'=' * 72}")
        print("MODEL ARTEFACTS")
        print("=" * 72)

        missing = [a for a in _MODEL_ARTEFACTS if not (args.model_dir / a).exists()]
        if missing:
            print(f"  MISSING: {', '.join(missing)}")
            model_result["artefacts_ok"] = False
        else:
            ns = json.loads((args.model_dir / "norm_stats.json").read_text(encoding="utf-8"))
            norm_len = len(ns.get("mean", []))
            if norm_len != _EXPECTED_NORM_LEN:
                print(
                    f"  WARNING: norm_stats.json has {norm_len} features "
                    f"(expected {_EXPECTED_NORM_LEN})"
                )
                model_result["artefacts_ok"] = False
            else:
                print(f"  OK — all artefacts present, norm_stats has {norm_len} features.")
                model_result["artefacts_ok"] = True

            model_result["norm_n_features"] = norm_len
            norm_mean = np.asarray(ns["mean"])
            norm_std  = np.asarray(ns["std"])
            norm_std[norm_std < 1e-12] = 1.0

            # Feature extraction for task XDFs
            print(f"\n{'=' * 72}")
            print("FEATURE CROSS-SESSION CHECK  (normalised vs calibration norm)")
            print("=" * 72)
            print(f"  {'Recording':<14}  {'Windows':>8}  {'Median |Z|':>10}  {'Mean |Z|':>9}")
            print(f"  {'-' * 50}")

            feat_means: dict[str, np.ndarray] = {}
            for acq in _FEATURE_ACQUISITIONS:
                path = xdf_paths.get(acq)
                if path is None:
                    print(f"  {acq:<14}  (not found)")
                    continue
                X_z = _extract_features_zscored(path, ch_names, norm_mean, norm_std)
                if X_z is None:
                    print(f"  {acq:<14}  (no windows extracted)")
                    continue
                feat_means[acq] = X_z.mean(axis=0)
                med_z  = float(np.median(np.abs(X_z)))
                mean_z = float(np.mean(np.abs(X_z)))
                model_result[f"{acq}_median_z"] = round(med_z,  2)
                model_result[f"{acq}_mean_z"]   = round(mean_z, 2)
                model_result[f"{acq}_n_windows"] = len(X_z)
                print(f"  {acq:<14}  {len(X_z):>8}  {med_z:>10.2f}  {mean_z:>9.2f}")

            # C1 vs C2 cross-run consistency
            if "cal_c1" in feat_means and "cal_c2" in feat_means:
                drift_c1c2 = float(np.abs(feat_means["cal_c1"] - feat_means["cal_c2"]).mean())
                model_result["c1_c2_drift_mean_z"] = round(drift_c1c2, 3)
                flag = "  <<< check" if drift_c1c2 > 1.0 else ""
                print(f"\n  C1-vs-C2 feature drift  (mean |ΔZ|): {drift_c1c2:.3f}{flag}")
                print( "    Expected: low (<0.5). High drift suggests cap shift between runs.")

            # Adaptation drift from calibration
            for cond in ("adaptation", "control"):
                if cond in feat_means and ("cal_c1" in feat_means or "cal_c2" in feat_means):
                    cal_keys = [k for k in ("cal_c1", "cal_c2") if k in feat_means]
                    cal_mean = np.mean([feat_means[k] for k in cal_keys], axis=0)
                    drift = float(np.abs(feat_means[cond] - cal_mean).mean())
                    model_result[f"{cond}_drift_from_cal_mean_z"] = round(drift, 3)
                    flag = "  <<< check" if drift > 1.5 else ""
                    print(
                        f"\n  {cond.capitalize()} drift from calibration (mean |ΔZ|): "
                        f"{drift:.3f}{flag}"
                    )
                    print("    Expected: moderate (0.3–1.0). Very high drift may indicate cap movement.")

    # ------------------------------------------------------------------
    # Write JSON summary
    # ------------------------------------------------------------------
    _QC_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _QC_DIR / f"{pid}_{session}_session_audit.json"

    summary = {
        "pid": pid,
        "session": session,
        "physio_dir": str(args.physio_dir),
        "model_dir": str(args.model_dir) if args.model_dir else None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": recording_results,
        "model": model_result,
        "overall_ok": not any_bad,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAudit summary saved: {json_path.relative_to(_REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Overall verdict
    # ------------------------------------------------------------------
    print()
    if not any_bad:
        print("  OVERALL: OK — all recordings pass signal quality checks.")
    else:
        print("  OVERALL: WARN — one or more recordings have issues (see above).")
    print()

    sys.exit(0 if not any_bad else 1)


if __name__ == "__main__":
    main()
