"""Study-day EEG signal check — run on rest baseline XDF before calibration.

Checks the 2-minute resting-state recording for flat/noisy channels and
timestamp regularity.  Prints a clear GO / NO-GO verdict and exits with
code 1 if any channels are flagged, so the operator can re-seat electrodes
before starting the (18-minute) calibration blocks.

Usage
-----
    python scripts/session/check_eeg_signal.py \\
        --xdf "C:/data/adaptive_matb/physiology/sub-P001/ses-S001/physio/\\
               sub-P001_ses-S001_task-matb_acq-rest_physio.xdf" \\
        [--out results/figures/P001/S001/eeg_signal_check.png]

Exit codes
----------
    0  no flagged channels (GO)
    1  one or more flat or noisy channels (NO-GO)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyxdf

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg import EegPreprocessor                          # noqa: E402
from eeg.xdf_loader import (                             # noqa: E402
    PREPROCESSING_CONFIG,
    _load_eeg_metadata,
    _merge_eeg_streams,
)

# ---------------------------------------------------------------------------
# Thresholds (all in µV)
# ---------------------------------------------------------------------------
_MIN_DURATION_S = 90.0     # warn if rest recording is shorter than this
_FLAT_UV        = 0.5      # preprocessed std below this → flat channel
_NOISY_UV       = 200.0    # preprocessed RMS above this → noisy channel
_TIMESTAMP_TOL  = 0.10     # 10 % tolerance on median inter-sample interval

# The ANT eego amplifier streams EEG data in volts via LSL.
_STREAM_TO_UV = 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_timestamps(ts: np.ndarray, target_srate: float) -> tuple[bool, str]:
    """Return (ok, message) for timestamp regularity of a (decimated) stream."""
    if len(ts) < 2:
        return True, "too few samples to check"
    diffs = np.diff(ts)
    expected = 1.0 / target_srate
    median_interval = float(np.median(diffs))
    n_gaps = int(np.sum(diffs > 2.0 * expected))
    max_gap_s = float(diffs.max())
    deviation = abs(median_interval - expected) / expected
    if deviation > _TIMESTAMP_TOL:
        return False, (
            f"median interval {median_interval * 1000:.2f} ms vs "
            f"expected {expected * 1000:.2f} ms  ({deviation * 100:.1f}% deviation)"
        )
    if n_gaps > 0:
        return False, f"{n_gaps} gap(s) detected (largest: {max_gap_s:.3f} s)"
    return True, f"median {median_interval * 1000:.2f} ms, {n_gaps} gaps"


def _load_and_decimate(xdf_path: Path) -> tuple[np.ndarray, np.ndarray, float, bool, int]:
    """Load XDF, merge EEG streams, decimate to target srate.

    Returns (raw_ts, eeg_ts, actual_srate, decimated, factor).
    Exits on fatal errors.
    """
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg_stream = _merge_eeg_streams(streams)
    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found in XDF.")

    raw_ts = np.array(eeg_stream["time_series"], dtype=np.float64).T  # (C, N)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Study-day EEG signal check.  Run on rest baseline XDF before calibration.  "
            "Exits 0 (GO) or 1 (NO-GO)."
        )
    )
    parser.add_argument(
        "--xdf", required=True, type=Path,
        help="Path to rest baseline .xdf recording.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Save per-channel RMS figure to this path. Defaults to <xdf_stem>_eeg_check.png alongside the XDF.",
    )
    args = parser.parse_args()

    if not args.xdf.exists():
        sys.exit(f"ERROR: XDF not found: {args.xdf}")

    if args.out is None:
        args.out = args.xdf.with_name(args.xdf.stem + "_eeg_check.png")

    # ------------------------------------------------------------------
    # Load & decimate
    # ------------------------------------------------------------------
    print(f"\nLoading {args.xdf.name} ...", flush=True)
    raw_ts, eeg_ts, actual_srate, decimated, factor = _load_and_decimate(args.xdf)

    n_ch, n_samples = raw_ts.shape
    target_srate = PREPROCESSING_CONFIG.srate
    duration_s = n_samples / target_srate

    ch_names = _load_eeg_metadata(_REPO_ROOT)
    if len(ch_names) != n_ch:
        print(
            f"WARNING: metadata has {len(ch_names)} channels but stream has {n_ch}. "
            "Using integer indices."
        )
        ch_names = [str(i) for i in range(n_ch)]

    print(f"  Channels : {n_ch}")
    if decimated:
        print(f"  Srate    : {actual_srate:.0f} Hz → {target_srate:.0f} Hz (decimated x{factor})")
    else:
        print(f"  Srate    : {actual_srate:.1f} Hz")
    print(f"  Duration : {duration_s:.1f} s  ({duration_s / 60:.1f} min)")

    if duration_s < _MIN_DURATION_S:
        print(
            f"\nWARNING: Duration {duration_s:.1f} s < {_MIN_DURATION_S:.0f} s minimum — "
            "recording may be incomplete."
        )

    # ------------------------------------------------------------------
    # Timestamp regularity
    # ------------------------------------------------------------------
    ts_ok, ts_msg = _check_timestamps(eeg_ts, target_srate)
    print(f"  Timestamps: {'OK' if ts_ok else 'WARNING'}  ({ts_msg})")

    # ------------------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------------------
    print("\nPreprocessing ...", flush=True)
    preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
    preprocessor.initialize_filters(n_ch, prewarm=raw_ts[:, 0])
    prep_ts = preprocessor.process(raw_ts)

    raw_rms  = np.sqrt(np.mean(raw_ts  ** 2, axis=1)) * _STREAM_TO_UV
    prep_rms = np.sqrt(np.mean(prep_ts ** 2, axis=1)) * _STREAM_TO_UV
    prep_std = prep_ts.std(axis=1) * _STREAM_TO_UV

    flat_idx  = [i for i in range(n_ch) if prep_std[i]  < _FLAT_UV]
    noisy_idx = [i for i in range(n_ch) if prep_rms[i] > _NOISY_UV]
    flagged_idx = sorted(set(flat_idx) | set(noisy_idx))

    # ------------------------------------------------------------------
    # Channel table
    # ------------------------------------------------------------------
    show_all = n_ch <= 32
    print(f"\n{'Channel':<8}  {'Raw RMS':>10}  {'Prep RMS':>10}  Status")
    print("-" * 46)
    printed_any = False
    for i, ch in enumerate(ch_names):
        if i in flat_idx:
            status = "FLAT"
        elif i in noisy_idx:
            status = "NOISY"
        else:
            status = ""
        if show_all or status:
            print(f"{ch:<8}  {raw_rms[i]:>8.1f} µV  {prep_rms[i]:>8.1f} µV  {status}")
            printed_any = True

    if not show_all and not printed_any:
        print("  (no flagged channels)")

    flat_names  = [ch_names[i] for i in flat_idx]
    noisy_names = [ch_names[i] for i in noisy_idx]
    print(f"\nFlat    ({len(flat_idx)}) : {', '.join(flat_names)  or 'none'}")
    print(f"Noisy   ({len(noisy_idx)}) : {', '.join(noisy_names) or 'none'}")
    print(f"Median prep RMS : {np.median(prep_rms):.1f} µV")

    # ------------------------------------------------------------------
    # Optional figure
    # ------------------------------------------------------------------
    if args.out is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(prep_rms)
            xs = np.arange(n_ch)
            colours = [
                "red"       if prep_rms[order[i]] > _NOISY_UV
                else "orange" if prep_std[order[i]] < _FLAT_UV
                else "steelblue"
                for i in range(n_ch)
            ]

            # Try to build a topomap panel using MNE + the repo channel locations.
            topo_ok = False
            try:
                import mne

                # Parse NA-271.elc manually — file lacks a Units header so
                # read_custom_montage cannot detect mm vs cm; do it ourselves.
                elc_path = _REPO_ROOT / "config" / "chanlocs" / "NA-271.elc"
                elc_pos: dict[str, tuple[float, float, float]] = {}
                with open(elc_path) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("NumberPositions") or line == "Positions":
                            continue
                        parts = line.replace(":", " ").split()
                        if len(parts) == 4:
                            elc_pos[parts[0]] = (
                                float(parts[1]) / 1000.0,  # mm → m
                                float(parts[2]) / 1000.0,
                                float(parts[3]) / 1000.0,
                            )

                # Keep only channels present in both the .elc and our recording.
                shared = [ch for ch in ch_names if ch in elc_pos]

                if len(shared) >= 4:
                    shared_idx = [ch_names.index(ch) for ch in shared]
                    rms_shared = prep_rms[shared_idx]

                    ch_pos_dict = {ch: np.array(elc_pos[ch]) for ch in shared}
                    montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict, coord_frame="unknown")
                    info = mne.create_info(ch_names=shared, sfreq=1.0, ch_types="eeg")
                    info.set_montage(montage, on_missing="ignore")

                    fig, axes = plt.subplots(
                        1, 3,
                        figsize=(18, 4),
                        gridspec_kw={"width_ratios": [5, 5, 3]},
                    )
                    topo_im, _ = mne.viz.plot_topomap(
                        rms_shared,
                        info,
                        axes=axes[2],
                        show=False,
                        cmap="RdYlGn_r",
                        vlim=(0, _NOISY_UV),
                        sensors=True,
                        contours=4,
                    )
                    plt.colorbar(topo_im, ax=axes[2], shrink=0.8, label="Prep RMS (µV)")
                    axes[2].set_title("Prep RMS topography")
                    topo_ok = True

            except Exception as _topo_err:
                pass  # fall through to 2-panel layout

            if not topo_ok:
                fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

            axes[0].bar(xs, raw_rms[order],  color=colours, alpha=0.8)
            axes[0].set_title("Raw RMS (µV)")
            axes[0].set_xlabel("Channel (sorted by prep RMS)")
            axes[0].set_ylabel("RMS (µV)")
            axes[1].bar(xs, prep_rms[order], color=colours, alpha=0.8)
            axes[1].set_title("Preprocessed RMS (µV)")
            axes[1].set_xlabel("Channel (sorted by prep RMS)")

            ts_label = "timestamps OK" if ts_ok else "TIMESTAMP WARNING"
            fig.suptitle(
                f"EEG Signal Check — {args.xdf.name}\n"
                f"Duration: {duration_s:.0f} s  |  Flat: {len(flat_idx)}  |  "
                f"Noisy: {len(noisy_idx)}  |  Median: {np.median(prep_rms):.1f} µV  |  "
                f"{ts_label}",
                fontsize=9,
            )
            fig.tight_layout()
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, dpi=150)
            plt.close(fig)
            print(f"\nFigure saved: {args.out}")
        except ImportError:
            print("WARNING: matplotlib not available — skipping figure.")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    if not ts_ok:
        print(
            "\nWARNING: Timestamp irregularities detected — check USB connections "
            "before proceeding."
        )

    n_bad = len(flagged_idx)
    print()
    if n_bad == 0:
        print("  GO — cap looks clean. Proceed to calibration.")
    else:
        names_str = ", ".join([ch_names[i] for i in flagged_idx])
        print(f"  NO-GO — {n_bad} channel(s) flagged: {names_str}")
        print("  Re-seat the electrode(s) listed above before starting calibration.")
    print()

    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
