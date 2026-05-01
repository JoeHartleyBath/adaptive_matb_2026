"""Deep EEG sanity-check / diagnostic for a single XDF file.

Run:
    python scripts/analysis/eeg_deep_diagnostic.py \
        "C:/data/adaptive_matb/physiology/sub-PDRY07/ses-S001/physio/sub-PDRY07_ses-S001_task-matb_acq-rest_physio.xdf" \
        [--compare "C:/data/.../sub-PDRY06/...rest_physio.xdf"] \
        [--out results/qc/deep_diag_PDRY07_rest.txt]

Tests / reports
---------------
  0. XDF stream header   : reported units, scale, amplifier info
  1. Raw-signal checks   : pre-filter, pre-CAR per-channel RMS (µV) distribution
  2. Scale-unit probe    : inferred data unit from raw RMS percentiles
  3. Filter-only checks  : post-bandpass/notch, pre-CAR per-channel RMS
  4. CAR impact check    : full-pipeline minus filter-only, channel-wise
  5. PSD of key channels : worst, median, and best channels (printed stats)
  6. Temporal stability  : rolling 10 s RMS (variance across epochs)
  7. Clipping check      : samples at ±hardware_limit
  8. Spatial pattern     : peripheral vs. central channel RMS comparison
  9. Comparison baseline : same metrics on a reference XDF (e.g. PDRY06)
 10. Summary verdict     : clear pass / marginal / fail per hypothesis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyxdf
from scipy import signal as spsig

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg import EegPreprocessor                          # noqa: E402
from eeg.eeg_preprocessing_config import EegPreprocessingConfig  # noqa: E402
from eeg.xdf_loader import (                             # noqa: E402
    PREPROCESSING_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)

# ---------------------------------------------------------------------------
_STREAM_TO_UV = 1e6   # ANT eego: streams in Volts
_TARGET_SRATE = 128.0

# Spatial groups: keys are label prefixes or explicit sets
# Approximate central vs. peripheral for 128-ch ANT cap
_CENTRAL_PREFIXES = ("Z", "C")
_PERIPHERAL_PREFIXES = ("LL", "RR", "LA", "LB", "LC", "LD", "LE",
                        "RA", "RB", "RC", "RD", "RE", "Lm", "RM")


# ---------------------------------------------------------------------------
def _load_raw(xdf_path: Path):
    """Return (raw_ts_V, srate_actual, ch_names) or raise."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        raise RuntimeError("No EEG stream found.")

    # Extract stream-level unit info if available
    try:
        ch_desc = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        units = [c.get("unit", ["?"])[0] for c in ch_desc]
    except Exception:
        units = ["?"]

    raw = np.array(eeg_stream["time_series"], dtype=np.float64).T  # (ch, samp)
    ts  = np.array(eeg_stream["time_stamps"])

    actual_srate = (
        (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1
        else float(eeg_stream["info"]["nominal_srate"][0])
    )

    # Decimate to target srate
    if actual_srate > _TARGET_SRATE * 1.1:
        factor = int(round(actual_srate / actual_srate * (_TARGET_SRATE / actual_srate * actual_srate)))
        # Simpler: target / actual * actual = target
        factor = int(round(actual_srate / _TARGET_SRATE))
        raw = raw[:, ::factor]
        ts  = ts[::factor]

    ch_names = _load_eeg_metadata(_REPO_ROOT)
    return raw, actual_srate, ch_names, units, eeg_stream["info"]


def _apply_bp_notch(raw: np.ndarray, n_ch: int) -> np.ndarray:
    """Bandpass + notch only (no CAR)."""
    cfg = EegPreprocessingConfig(
        bp_low_hz=0.5, bp_high_hz=40.0, bp_order=4,
        notch_freq=50.0, notch_quality=30.0,
        apply_car=False, srate=_TARGET_SRATE,
    )
    p = EegPreprocessor(cfg)
    p.initialize_filters(n_ch)
    return p.process(raw)


def _apply_car(filtered: np.ndarray) -> np.ndarray:
    """CAR only."""
    return filtered - filtered.mean(axis=0)


def _rms_uv(data: np.ndarray) -> np.ndarray:
    """Per-channel RMS in µV. data: (n_ch, n_samp)."""
    return np.sqrt(np.mean(data ** 2, axis=1)) * _STREAM_TO_UV


def _psd_db(data_1ch: np.ndarray, srate: float):
    """Return (freqs, psd_dB) for a single channel."""
    f, pxx = spsig.welch(data_1ch, fs=srate, nperseg=int(srate * 4))
    return f, 10 * np.log10(np.maximum(pxx, 1e-30))


def _spatial_group(ch_names):
    """Return (central_idx, peripheral_idx)."""
    central, peripheral = [], []
    for i, ch in enumerate(ch_names):
        if any(ch.startswith(p) for p in _PERIPHERAL_PREFIXES):
            peripheral.append(i)
        elif any(ch.startswith(p) for p in _CENTRAL_PREFIXES):
            central.append(i)
    return central, peripheral


def _rolling_rms_uv(data: np.ndarray, srate: float, window_s: float = 10.0) -> np.ndarray:
    """Return (n_ch, n_epochs) rolling epoch RMS in µV."""
    w = int(srate * window_s)
    n_ch, n_samp = data.shape
    n_epochs = n_samp // w
    if n_epochs == 0:
        return _rms_uv(data).reshape(-1, 1)
    epochs = data[:, :n_epochs * w].reshape(n_ch, n_epochs, w)
    return np.sqrt(np.mean(epochs ** 2, axis=2)) * _STREAM_TO_UV


def _report_separator(title: str, out):
    w = 72
    out.write("\n" + "=" * w + "\n")
    out.write(f"  {title}\n")
    out.write("=" * w + "\n")


def _print_percentile_table(label: str, vals: np.ndarray, out):
    pcts = [0, 5, 25, 50, 75, 95, 100]
    row = "  ".join(f"p{p:3d}={np.percentile(vals, p):7.1f}" for p in pcts)
    out.write(f"  {label:<30s}  {row}\n")


def run_diagnostics(
    xdf_path: Path,
    compare_path: Path | None,
    out_path: Path | None,
):
    output_lines: list[str] = []

    class _Tee:
        def write(self, s):
            sys.stdout.write(s)
            output_lines.append(s)
        def flush(self):
            sys.stdout.flush()

    out = _Tee()

    # ------------------------------------------------------------------
    out.write(f"EEG Deep Diagnostic\n")
    out.write(f"File : {xdf_path}\n")
    out.write(f"Date : {__import__('datetime').datetime.now().isoformat()}\n")

    # ------------------------------------------------------------------
    _report_separator("0. XDF STREAM HEADER", out)
    raw, actual_srate, ch_names, units, info = _load_raw(xdf_path)
    n_ch, n_samp = raw.shape
    dur_s = n_samp / _TARGET_SRATE

    out.write(f"  Channels        : {n_ch}\n")
    out.write(f"  Samples         : {n_samp}  ({dur_s:.1f} s at {_TARGET_SRATE} Hz after decimation)\n")
    out.write(f"  Actual srate    : {actual_srate:.1f} Hz\n")
    out.write(f"  Nominal srate   : {info.get('nominal_srate', ['?'])[0]} Hz\n")
    manufacturer = info.get("manufacturer", ["?"])[0] if info.get("manufacturer") else "?"
    out.write(f"  Manufacturer    : {manufacturer}\n")
    amp_model = info.get("name", ["?"])[0] if info.get("name") else "?"
    out.write(f"  Stream name     : {amp_model}\n")
    unique_units = list(dict.fromkeys(units)) if units else ["?"]
    out.write(f"  Channel units (from desc): {unique_units[:5]}{'...' if len(unique_units) > 5 else ''}\n")

    # ------------------------------------------------------------------
    _report_separator("1. RAW SIGNAL (pre-filter, pre-CAR)", out)
    raw_rms = _rms_uv(raw)
    out.write(f"  Per-channel RMS distribution (µV):\n")
    _print_percentile_table("All channels", raw_rms, out)

    # ------------------------------------------------------------------
    _report_separator("2. SCALE / UNIT PROBE", out)
    med_raw_uv = float(np.median(raw_rms))
    out.write(f"  Median raw RMS (×1e6)         : {med_raw_uv:.1f} µV\n")
    out.write(f"  Median raw RMS (as-is, no ×)  : {np.median(np.sqrt(np.mean(raw**2, axis=1))):.6f}\n")
    if med_raw_uv < 1:
        verdict = "VERY SUSPICIOUS — values suggest data is NOT in volts (already µV or nV?)."
    elif med_raw_uv < 50:
        verdict = "OK — consistent with data in volts (EEG ~1-50 µV after filter)."
    elif med_raw_uv < 300:
        verdict = "ELEVATED but plausible — raw EEG unfiltered or contains slow drift."
    else:
        verdict = "VERY HIGH — likely artifact, scaling issue, or saturation."
    out.write(f"  Verdict: {verdict}\n")

    # ------------------------------------------------------------------
    _report_separator("3. FILTER-ONLY (bandpass + notch, NO CAR)", out)
    filt = _apply_bp_notch(raw, n_ch)
    filt_rms = _rms_uv(filt)
    out.write(f"  Per-channel RMS after bandpass+notch (µV):\n")
    _print_percentile_table("All channels", filt_rms, out)
    _print_percentile_table("Filter improvement (raw/filt)", raw_rms / np.maximum(filt_rms, 1e-9), out)

    noisy_filt = [ch_names[i] for i in range(n_ch) if filt_rms[i] > 200.0]
    out.write(f"\n  Channels >200 µV after filter (pre-CAR): {len(noisy_filt)}\n")
    if noisy_filt:
        out.write(f"  {noisy_filt}\n")

    # ------------------------------------------------------------------
    _report_separator("4. FULL PIPELINE (bandpass + notch + CAR)", out)
    full = _apply_car(filt)
    full_rms = _rms_uv(full)
    out.write(f"  Per-channel RMS after full pipeline (µV):\n")
    _print_percentile_table("All channels", full_rms, out)

    noisy_full = [ch_names[i] for i in range(n_ch) if full_rms[i] > 200.0]
    out.write(f"\n  Channels >200 µV after full pipeline: {len(noisy_full)}\n")
    if noisy_full:
        out.write(f"  {noisy_full}\n")

    car_delta = full_rms - filt_rms
    out.write(f"\n  CAR impact (full - filter_only) per-channel RMS:\n")
    _print_percentile_table("Δ RMS (µV)", car_delta, out)
    if car_delta.mean() > 10:
        out.write("  >> CAR is INCREASING average channel noise — suggests bad channels are contaminating the reference.\n")
    elif car_delta.mean() < -10:
        out.write("  >> CAR is REDUCING average channel noise — functioning as intended.\n")
    else:
        out.write("  >> CAR has minimal net effect.\n")

    # ------------------------------------------------------------------
    _report_separator("5. PSD OF KEY CHANNELS", out)
    sorted_full = np.argsort(full_rms)
    key_idx = {
        "best (lowest RMS)"  : sorted_full[0],
        "p25"                : sorted_full[n_ch // 4],
        "median"             : sorted_full[n_ch // 2],
        "p75"                : sorted_full[3 * n_ch // 4],
        "worst (highest RMS)": sorted_full[-1],
    }
    out.write(f"  {'Channel':<20s}  {'Name':<8s}  {'RMS µV':>8s}  {'Delta+1 Hz':>12s}  {'10-20 Hz':>10s}  {'30-40 Hz':>10s}  {'40-48 Hz':>10s}\n")
    out.write(f"  {'-'*90}\n")

    band_ranges = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40), (40, 48)]

    for label, idx in key_idx.items():
        ch_data = full[idx]
        f, pdb = _psd_db(ch_data, _TARGET_SRATE)
        def band_power(flo, fhi):
            mask = (f >= flo) & (f < fhi)
            return float(np.mean(pdb[mask])) if mask.any() else float("nan")
        p10_20 = band_power(10, 20)
        p30_40 = band_power(30, 40)
        p40_48 = band_power(40, 48)
        p1     = band_power(1, 4)
        out.write(f"  {label:<20s}  {ch_names[idx]:<8s}  {full_rms[idx]:8.1f}  {p1:12.1f}  {p10_20:10.1f}  {p30_40:10.1f}  {p40_48:10.1f}\n")

    # Broad spectral summary for ALL channels
    out.write(f"\n  Median across ALL channels:\n")
    band_medians = {}
    for flo, fhi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]:
        powers = []
        for i in range(n_ch):
            f_ch, p_ch = _psd_db(full[i], _TARGET_SRATE)
            mask = (f_ch >= flo) & (f_ch < fhi)
            if mask.any():
                powers.append(np.mean(p_ch[mask]))
        med = float(np.median(powers))
        band_medians[f"{flo}-{fhi}Hz"] = med
        out.write(f"    {flo:2d}-{fhi:2d} Hz : {med:6.1f} dB\n")

    # ------------------------------------------------------------------
    _report_separator("6. TEMPORAL STABILITY (10-s rolling epoch RMS)", out)
    rolling = _rolling_rms_uv(full, _TARGET_SRATE, window_s=10.0)
    n_epochs = rolling.shape[1]
    out.write(f"  Epochs: {n_epochs}\n")
    if n_epochs > 1:
        epoch_median = np.median(rolling, axis=0)
        out.write(f"  Median-channel RMS per 10-s epoch (µV): "
                  f"{' '.join(f'{v:.0f}' for v in epoch_median)}\n")
        cv = epoch_median.std() / (epoch_median.mean() + 1e-9)
        out.write(f"  Epoch-to-epoch CV of median RMS: {cv:.3f} "
                  f"({'STABLE' if cv < 0.2 else 'VARIABLE'})\n")

        # Detect whether noise decreases over time (filter transient?)
        if epoch_median[-1] < epoch_median[0] * 0.7:
            out.write("  >> Noise DECREASES across recording — filter transient or settling artefact at start.\n")
        elif epoch_median[-1] > epoch_median[0] * 1.3:
            out.write("  >> Noise INCREASES across recording — participant movement or fatigue?\n")
        else:
            out.write("  >> Noise is roughly stationary across recording.\n")

    # ------------------------------------------------------------------
    _report_separator("7. CLIPPING / SATURATION CHECK", out)
    abs_max = float(np.abs(raw).max())
    abs_max_uv = abs_max * _STREAM_TO_UV
    percentile_999 = float(np.percentile(np.abs(raw), 99.9)) * _STREAM_TO_UV
    out.write(f"  Absolute max across all channels : {abs_max_uv:.1f} µV\n")
    out.write(f"  99.9th percentile                : {percentile_999:.1f} µV\n")
    # Check for clusters of identical values (clipping signature)
    n_saturated = 0
    for i in range(n_ch):
        ch = raw[i]
        diff = np.diff(ch)
        # >5 consecutive identical samples likely clipping
        runs = np.where(diff == 0)[0]
        if len(runs) > 5:
            n_saturated += 1
    out.write(f"  Channels with >5 consecutive identical samples: {n_saturated}\n")
    if abs_max_uv > 800:
        out.write("  >> Values exceed ±800 µV — possible saturation/clipping.\n")
    else:
        out.write("  >> No obvious hard clipping detected.\n")

    # ------------------------------------------------------------------
    _report_separator("6b. SETTLED SIGNAL QUALITY (skip first 20 s)", out)
    skip_samples = int(20.0 * _TARGET_SRATE)
    if n_samp > skip_samples + int(10.0 * _TARGET_SRATE):
        settled = full[:, skip_samples:]
        settled_rms = _rms_uv(settled)
        out.write(f"  Skipping first {skip_samples} samples (20 s) to bypass filter transient.\n")
        out.write(f"  Settled signal per-channel RMS (µV):\n")
        _print_percentile_table("All channels (settled)", settled_rms, out)
        noisy_settled = [ch_names[i] for i in range(n_ch) if settled_rms[i] > 200.0]
        warn_settled  = [ch_names[i] for i in range(n_ch) if 100 < settled_rms[i] <= 200.0]
        ok_settled    = [ch_names[i] for i in range(n_ch) if settled_rms[i] <= 100.0]
        out.write(f"\n  After settling — channels >200 µV (NOISY) : {len(noisy_settled)}\n")
        if noisy_settled:
            out.write(f"  {noisy_settled}\n")
        out.write(f"  After settling — channels 100-200 µV (warn): {len(warn_settled)}\n")
        out.write(f"  After settling — channels ≤100 µV (ok)      : {len(ok_settled)}\n")
        if len(noisy_settled) < len(noisy_full):
            out.write(f"\n  >> Dropping from {len(noisy_full)} to {len(noisy_settled)} noisy channels after "
                      f"settling — audit noise count is inflated by filter transient.\n")
    else:
        out.write("  Recording too short to compute settled-state metrics.\n")
        settled_rms = full_rms  # fallback

    # ------------------------------------------------------------------
    _report_separator("8. SPATIAL PATTERN (central vs. peripheral)", out)
    central_idx, peripheral_idx = _spatial_group(ch_names)
    mid_idx = [i for i in range(n_ch) if i not in central_idx and i not in peripheral_idx]

    for group_label, idx_list in [
        ("Central  (Z, C-prefixed)", central_idx),
        ("Peripheral (LL/RR/LA-E/RA-E/Lm/RM)", peripheral_idx),
        ("Other / unclassified", mid_idx),
    ]:
        if idx_list:
            rms_group = full_rms[idx_list]
            n_bad = int(np.sum(rms_group > 200))
            out.write(f"  {group_label:<38s}  n={len(idx_list):3d}  "
                      f"median RMS={np.median(rms_group):6.1f} µV  "
                      f"bad(>200)={n_bad}/{len(idx_list)}\n")

    # Check if bad channels cluster on periphery
    if peripheral_idx:
        pct_peripheral_bad = sum(1 for i in peripheral_idx if full_rms[i] > 200) / len(peripheral_idx)
        pct_central_bad    = (sum(1 for i in central_idx if full_rms[i] > 200) / len(central_idx)
                              if central_idx else 0.0)
        out.write(f"\n  Fraction bad: peripheral={pct_peripheral_bad:.0%}  central={pct_central_bad:.0%}\n")
        if pct_peripheral_bad > 2 * pct_central_bad + 0.1:
            out.write("  >> Bad channels are PREDOMINANTLY PERIPHERAL — consistent with cap fit / electrode contact issue.\n")
        elif pct_central_bad > 0.3:
            out.write("  >> Central channels also heavily affected — suggests global noise (reference, movement, power).\n")
        else:
            out.write("  >> Spatial pattern is mixed.\n")

    # ------------------------------------------------------------------
    _report_separator("9. PER-CHANNEL RMS TABLE (full pipeline)", out)
    out.write(f"  {'Ch':<8s}  {'RMS µV':>8s}  {'Flag':>6s}\n")
    out.write(f"  {'-'*30}\n")
    for i in range(n_ch):
        flag = "NOISY" if full_rms[i] > 200 else ("warn" if full_rms[i] > 100 else "ok")
        out.write(f"  {ch_names[i]:<8s}  {full_rms[i]:8.1f}  {flag:>6s}\n")

    # ------------------------------------------------------------------
    if compare_path is not None:
        _report_separator("10. COMPARISON BASELINE", out)
        try:
            raw_c, srate_c, ch_c, units_c, _ = _load_raw(compare_path)
            n_ch_c = raw_c.shape[0]
            filt_c = _apply_bp_notch(raw_c, n_ch_c)
            full_c = _apply_car(filt_c)
            full_rms_c = _rms_uv(full_c)
            noisy_c = int(np.sum(full_rms_c > 200))
            out.write(f"  Reference: {compare_path.name}\n")
            out.write(f"  Channels  : {n_ch_c}\n")
            _print_percentile_table("Ref full-pipeline RMS (µV)", full_rms_c, out)
            _print_percentile_table("PDRY07 full-pipeline RMS (µV)", full_rms, out)
            out.write(f"\n  Noisy channels (>200 µV) — reference: {noisy_c}  target: {len(noisy_full)}\n")
            delta_med = float(np.median(full_rms)) - float(np.median(full_rms_c))
            out.write(f"  Median RMS delta (target - reference): {delta_med:+.1f} µV\n")
            if abs(delta_med) < 20:
                out.write("  >> Recordings are COMPARABLE in overall noise level.\n")
            elif delta_med > 0:
                out.write("  >> Target is NOISIER than reference by more than 20 µV.\n")
            else:
                out.write("  >> Target is CLEANER than reference by more than 20 µV.\n")
        except Exception as e:
            out.write(f"  ERROR loading reference: {e}\n")

    # ------------------------------------------------------------------
    _report_separator("SUMMARY VERDICT", out)
    issues = []
    positives = []

    if med_raw_uv < 50:
        positives.append("Raw signal amplitude is normal for EEG-in-volts.")
    elif med_raw_uv > 300:
        issues.append("Raw RMS is extremely high — check amplifier scale/units.")

    if len(noisy_filt) < 20:
        positives.append(f"Only {len(noisy_filt)} channels noisy BEFORE CAR — signal itself may be OK.")
    elif len(noisy_filt) >= 40:
        issues.append(f"{len(noisy_filt)} channels >200 µV even before CAR.")

    if len(noisy_full) > len(noisy_filt):
        issues.append("CAR is WORSENING the noise count — bad channels contaminating reference.")
    elif len(noisy_full) < len(noisy_filt):
        positives.append("CAR is reducing noisy channel count — functioning correctly.")

    if n_epochs > 1:
        cv_val = np.std(np.median(rolling, axis=0)) / (np.mean(np.median(rolling, axis=0)) + 1e-9)
        if cv_val < 0.2:
            positives.append("Signal is temporally stable (low epoch-to-epoch variance).")
        else:
            issues.append("Temporal instability detected — recording may contain transient artefacts.")

    if n_saturated == 0:
        positives.append("No clipping / saturation detected.")
    else:
        issues.append(f"{n_saturated} channels show repeated identical samples (possible clipping).")

    if peripheral_idx and central_idx:
        pct_p = sum(1 for i in peripheral_idx if full_rms[i] > 200) / len(peripheral_idx)
        pct_c = sum(1 for i in central_idx if full_rms[i] > 200) / len(central_idx)
        if pct_p > 0.5 and pct_c < 0.3:
            issues.append("Noise is predominantly peripheral — cap contact issue but central channels may be usable.")
        elif pct_c > 0.3:
            issues.append("Central channels are also noisy — systemic noise problem.")
        else:
            positives.append("Central channels are mostly clean.")

    out.write("\n  ISSUES DETECTED:\n")
    for msg in issues:
        out.write(f"    [!] {msg}\n")
    if not issues:
        out.write("    None.\n")

    out.write("\n  POSITIVE INDICATORS:\n")
    for msg in positives:
        out.write(f"    [+] {msg}\n")
    if not positives:
        out.write("    None.\n")

    # ------------------------------------------------------------------
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(output_lines), encoding="utf-8")
        print(f"\n[Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Deep EEG diagnostic for a single XDF.")
    ap.add_argument("xdf", type=Path, help="Target XDF file path.")
    ap.add_argument("--compare", type=Path, default=None,
                    help="Optional reference XDF (e.g. PDRY06 rest) for side-by-side comparison.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional output text file path.")
    args = ap.parse_args()

    if not args.xdf.exists():
        print(f"ERROR: file not found: {args.xdf}", file=sys.stderr)
        sys.exit(1)

    run_diagnostics(args.xdf, args.compare, args.out)


if __name__ == "__main__":
    main()
