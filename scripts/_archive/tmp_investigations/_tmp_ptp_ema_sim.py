"""Offline simulation: per-channel PTP gate + slow-EMA drift correction.

Runs offline inference on control + adaptation for two participants under
4 pipeline variants:

    baseline  — existing pipeline, no modifications
    +PTP      — per-channel amplitude gate; hold-last-good on bad windows
    +EMA      — slow-EMA feature drift correction (τ ≈ 5 min at 4 Hz)
    +Both     — PTP gate AND EMA correction combined

Both participants are plotted side-by-side in a single figure.

Output:
    results/figures/ptp_ema_sim_mwl.png

Run:
    .venv/Scripts/python.exe scripts/_tmp_ptp_ema_sim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor
from eeg.online_features import OnlineFeatureExtractor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE     = 128.0
WINDOW_S  = 2.0
STEP_S    = 0.25
WIN_SAMP  = int(WINDOW_S * SRATE)   # 256
STEP_SAMP = int(STEP_S   * SRATE)   # 32

# Per-channel PTP threshold.
# ANT eego streams in Volts. Post-filtering (BP 0.5-40 Hz + CAR), typical
# clean EEG has per-channel 2s PTP of ~100-200 µV (P50 ≈ 120-160 µV, P99
# ≈ 310-340 µV from empirical measurement on PDRY06/PSELF data).
# We want to catch obvious artifacts (blinks: >500 µV on frontal channels;
# electrode pops: >1 mV on 1-2 channels; sustained EMG: >500 µV on many ch).
# 500 µV = 5e-4 V sits above the P99 of clean EEG and below typical
# artifact amplitude.  Combined with a fraction-based gate (>5% of channels)
# to avoid triggering on single-channel noise.
PTP_THRESHOLD_V  = 5e-4   # 500 µV — per-channel amplitude threshold
PTP_BAD_CH_FRAC  = 0.05   # flag if >5% of channels (>~6/128) exceed threshold

# Hold-last-good: reset to 0.5 after this many consecutive bad windows (5 s)
HOLD_RESET_N = 20

# Slow-EMA time constant.
# τ = 1/DRIFT_ALPHA steps × 0.25 s/step.
# τ=5 min (1/1200) was too fast — it tracked within-session workload cycles
# (~1-3 min blocks) and removed the workload signal along with drift.
# τ=20 min (1/4800) is slow enough to ignore workload cycles while still
# correcting genuine hardware / impedance drift over a 30-60 min session.
DRIFT_ALPHA = 1.0 / 4800.0   # τ ≈ 20 min at 4 Hz

# Warm-up: do not apply EMA correction until this many windows have been seen
EMA_WARMUP_N = 120   # 30 s at 4 Hz — EMA init phase only

_DATA_ROOT = Path(r"C:\data\adaptive_matb")
META_PATH  = _REPO / "config" / "eeg_metadata.yaml"
FEAT_CFG   = _REPO / "config" / "eeg_feature_extraction.yaml"
OUT_FIG    = _REPO / "results" / "figures" / "ptp_ema_sim_mwl.png"

PARTICIPANTS = [
    dict(pid="PDRY06", sid="S001"),
    dict(pid="PSELF",  sid="S005"),
]

SIM_CONDITIONS = ["control", "adaptation"]

# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------
def load_and_filter(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XDF, decimate to SRATE, apply EegPreprocessor.
    Returns (C×N float64, timestamps).
    """
    streams, _ = pyxdf.load_xdf(str(path))
    eeg  = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac  = int(round(actual / SRATE))
            data = data[:, ::fac]
            ts   = ts[::fac]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    return pp.process(data).astype(np.float64), ts


# ---------------------------------------------------------------------------
# Window quality: fraction-based per-channel PTP gate
# ---------------------------------------------------------------------------
def _window_ok(window: np.ndarray) -> tuple[bool, int]:
    """Return (is_good, n_bad_channels) using fraction-based PTP gate.

    Flags a window if >PTP_BAD_CH_FRAC of channels exceed PTP_THRESHOLD_V.
    A fraction-based gate avoids the 'any single channel' tripwire problem:
    with 128 channels, even ~5% of channels individually exceeding a too-low
    threshold makes P(at least one flagged) ≈ 100%.
    """
    ptp   = np.ptp(window, axis=1)                  # (C,)
    n_bad = int((ptp > PTP_THRESHOLD_V).sum())
    frac  = n_bad / window.shape[0]
    return frac <= PTP_BAD_CH_FRAC, n_bad


def _ptp_percentiles(data: np.ndarray) -> dict:
    """Compute P1/P25/P50/P75/P99/max of per-channel PTP over all 2s windows."""
    n = data.shape[1]
    all_ptp: list[np.ndarray] = []
    for s in range(WIN_SAMP, n - STEP_SAMP, STEP_SAMP):
        all_ptp.append(np.ptp(data[:, s - WIN_SAMP : s], axis=1))
    A = np.concatenate(all_ptp)
    ps = np.percentile(A, [1, 25, 50, 75, 99])
    return dict(p1=ps[0], p25=ps[1], p50=ps[2], p75=ps[3], p99=ps[4], mx=A.max())


# ---------------------------------------------------------------------------
# Per-participant inference
# ---------------------------------------------------------------------------
def run_participant(pid: str, sid: str) -> dict[str, dict]:
    """Load model + XDFs for one participant; run 4-variant inference.

    Returns {condition: result_dict} where result_dict has keys:
        t, baseline, ptp, ema, both, ptp_flags, n_bad_ch
    """
    model_dir = _DATA_ROOT / "models" / pid
    physio    = _DATA_ROOT / "physiology" / f"sub-{pid}" / f"ses-{sid}" / "physio"
    pfx       = f"sub-{pid}_ses-{sid}_task-matb_acq"

    # Load model artefacts
    pipeline = joblib.load(model_dir / "pipeline.pkl")
    selector = joblib.load(model_dir / "selector.pkl")
    with open(model_dir / "norm_stats.json") as f:
        ns = json.load(f)
    with open(model_dir / "model_config.json") as f:
        model_cfg = json.load(f)
    norm_mean  = np.array(ns["mean"], dtype=np.float64)
    norm_std   = np.array(ns["std"],  dtype=np.float64)
    norm_std[norm_std < 1e-12] = 1.0
    n_classes  = int(ns.get("n_classes", 3))
    p_high_col = n_classes - 1
    threshold  = float(model_cfg["youden_threshold"])

    meta     = yaml.safe_load(META_PATH.read_text())
    ch_names = meta["channel_names"]
    extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=FEAT_CFG)
    extractor.compute(np.zeros((len(ch_names), WIN_SAMP)))  # warm-up

    def _classify(win: np.ndarray, nm: np.ndarray, ns2: np.ndarray) -> float:
        feats     = extractor.compute(win)
        feats_z   = (feats - nm) / ns2
        feats_sel = selector.transform(feats_z[np.newaxis, :])
        return float(pipeline.predict_proba(feats_sel)[0][p_high_col])

    participant_results: dict[str, dict] = {}

    for cond in SIM_CONDITIONS:
        path = physio / f"{pfx}-{cond}_physio.xdf"
        if not path.exists():
            print(f"  [{pid}] {cond}: NOT FOUND — skipping")
            continue

        print(f"  [{pid} {sid}] {cond} ...", flush=True)
        data, ts = load_and_filter(path)
        n = data.shape[1]

        # Diagnostic: print PTP distribution to verify threshold is sensible
        pts = _ptp_percentiles(data)
        print(f"    PTP (V) P1={pts['p1']:.2e}  P50={pts['p50']:.2e}  "
              f"P99={pts['p99']:.2e}  max={pts['mx']:.2e}  "
              f"(gate: >{PTP_THRESHOLD_V:.0e} V on >{PTP_BAD_CH_FRAC*100:.0f}% ch)")

        t_vals     = []
        p_baseline = []
        p_ptp      = []
        p_ema      = []
        p_both     = []
        ptp_flags  = []
        n_bad_ch   = []

        last_good_ptp  = 0.5
        consec_bad_ptp = 0
        last_good_both = 0.5
        consec_bad_both = 0

        # Slow-EMA state: initialised from norm_mean (calibration baseline).
        # Starting from norm_mean means the correction is 0 at session start
        # and only grows if the long-run feature mean drifts away from
        # calibration.  Do NOT init from first window — that would apply the
        # full calibration→task shift as a correction immediately.
        ema_state: np.ndarray = norm_mean.copy()
        ema_count = 0

        for start in range(WIN_SAMP, n - STEP_SAMP, STEP_SAMP):
            window = data[:, start - WIN_SAMP : start]

            # --- PTP quality gate ---
            is_good, n_bad = _window_ok(window)
            ptp_flags.append(0 if is_good else 1)
            n_bad_ch.append(n_bad)

            # --- Baseline (no modifications) ---
            p_b = _classify(window, norm_mean, norm_std)
            p_baseline.append(p_b)

            # --- +PTP: hold-last-good ---
            if is_good:
                last_good_ptp  = p_b
                consec_bad_ptp = 0
                p_ptp.append(p_b)
            else:
                consec_bad_ptp += 1
                if consec_bad_ptp >= HOLD_RESET_N:
                    last_good_ptp = 0.5
                p_ptp.append(last_good_ptp)

            # --- Slow-EMA state update (on every window, artifact or not) ---
            raw_feats = extractor.compute(window)
            ema_state = DRIFT_ALPHA * raw_feats + (1.0 - DRIFT_ALPHA) * ema_state
            ema_count += 1

            # --- +EMA: apply correction after warm-up ---
            # Correction = raw_feats - (ema_state - norm_mean)
            # = removes how far the slow running mean has drifted from
            #   the calibration baseline, while preserving workload variance.
            if ema_count >= EMA_WARMUP_N:
                feats_corrected = raw_feats - (ema_state - norm_mean)
                feats_z   = (feats_corrected - norm_mean) / norm_std
                feats_sel = selector.transform(feats_z[np.newaxis, :])
                p_e = float(pipeline.predict_proba(feats_sel)[0][p_high_col])
            else:
                p_e = p_b   # before warm-up, same as baseline
            p_ema.append(p_e)

            # --- +Both: PTP gate over EMA-corrected P ---
            if is_good:
                last_good_both  = p_e
                consec_bad_both = 0
                p_both.append(p_e)
            else:
                consec_bad_both += 1
                if consec_bad_both >= HOLD_RESET_N:
                    last_good_both = 0.5
                p_both.append(last_good_both)

            t_vals.append(start / SRATE)

        t_arr     = np.array(t_vals)
        ptp_flags = np.array(ptp_flags, dtype=np.uint8)
        n_bad_arr = np.array(n_bad_ch,  dtype=np.int16)
        n_art     = int(ptp_flags.sum())
        pct       = 100.0 * n_art / len(ptp_flags) if len(ptp_flags) else 0.0

        print(f"    {len(ptp_flags)} windows | PTP flagged: {n_art} ({pct:.1f}%)")
        print(f"    mean bad channels (flagged only): "
              f"{n_bad_arr[ptp_flags == 1].mean():.1f}"
              if n_art > 0 else "    (no flagged windows)")
        print(f"    baseline mean P(HIGH) = {np.mean(p_baseline):.3f}")
        print(f"    +PTP     mean P(HIGH) = {np.mean(p_ptp):.3f}")
        print(f"    +EMA     mean P(HIGH) = {np.mean(p_ema):.3f}")
        print(f"    +Both    mean P(HIGH) = {np.mean(p_both):.3f}")

        participant_results[cond] = dict(
            t          = t_arr,
            baseline   = np.array(p_baseline),
            ptp        = np.array(p_ptp),
            ema        = np.array(p_ema),
            both       = np.array(p_both),
            ptp_flags  = ptp_flags,
            n_bad_ch   = n_bad_arr,
            threshold  = threshold,
            pid        = pid,
            sid        = sid,
        )

    return participant_results


# ---------------------------------------------------------------------------
# Run all participants
# ---------------------------------------------------------------------------
print("=" * 60)
print("PTP gate + slow-EMA simulation")
print(f"  PTP threshold   = {PTP_THRESHOLD_V:.0e} V ({PTP_THRESHOLD_V*1e6:.0f} µV) on >{PTP_BAD_CH_FRAC*100:.0f}% of channels")
print(f"  DRIFT_ALPHA     = 1/{int(1/DRIFT_ALPHA)}  (τ ≈ {1/DRIFT_ALPHA*STEP_S/60:.0f} min)  EMA init=norm_mean")
print(f"  EMA warmup      = {EMA_WARMUP_N} windows ({EMA_WARMUP_N * STEP_S:.0f} s)")
print("=" * 60)

all_results: dict[str, dict[str, dict]] = {}  # pid → cond → result
for p in PARTICIPANTS:
    pid, sid = p["pid"], p["sid"]
    print(f"\n{'─'*40}")
    print(f"Participant: {pid}  Session: {sid}")
    print(f"{'─'*40}")
    all_results[pid] = run_participant(pid, sid)


# ---------------------------------------------------------------------------
# Figure: 2 rows (conditions) × 2 columns (participants)
# ---------------------------------------------------------------------------
COLORS = {
    "baseline": "#888888",
    "ptp":      "#e74c3c",
    "ema":      "#2980b9",
    "both":     "#27ae60",
}
LABELS = {
    "baseline": "Baseline",
    "ptp":      "+PTP gate",
    "ema":      "+EMA drift",
    "both":     "+Both",
}

n_rows = len(SIM_CONDITIONS)
n_cols = len(PARTICIPANTS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(9 * n_cols, 4 * n_rows),
    constrained_layout=True,
    sharey="row",
)

fig.suptitle(
    "Per-channel PTP gate + slow-EMA drift correction — offline simulation\n"
    f"PTP: >{PTP_THRESHOLD_V*1e6:.0f} µV on >{PTP_BAD_CH_FRAC*100:.0f}% ch  |  "
    f"EMA τ ≈ {1/DRIFT_ALPHA*STEP_S/60:.0f} min, init=norm_mean  |  "
    "Red shading = PTP artifact",
    fontsize=11,
)

for row, cond in enumerate(SIM_CONDITIONS):
    for col, p in enumerate(PARTICIPANTS):
        pid = p["pid"]
        ax  = axes[row][col]

        if pid not in all_results or cond not in all_results[pid]:
            ax.set_visible(False)
            continue

        r   = all_results[pid][cond]
        t   = r["t"]
        thr = r["threshold"]

        # Shade artifact windows
        flags = r["ptp_flags"]
        i = 0
        while i < len(flags):
            if flags[i] == 1:
                j = i
                while j < len(flags) and flags[j] == 1:
                    j += 1
                ax.axvspan(t[i], t[j - 1], color="#e74c3c", alpha=0.15, lw=0)
                i = j
            else:
                i += 1

        # P(HIGH) lines
        for key in ("baseline", "ptp", "ema", "both"):
            lw  = 1.2 if key == "baseline" else 1.5
            ax.plot(t, r[key], color=COLORS[key], lw=lw,
                    alpha=0.85, label=LABELS[key])

        ax.axhline(thr, color="black", lw=0.9, ls="--", alpha=0.6,
                   label=f"Threshold ({thr:.3f})")

        n_art = int(flags.sum())
        pct   = 100.0 * n_art / len(flags) if len(flags) else 0.0
        ax.set_title(
            f"{pid} {r['sid']} — {cond.capitalize()}\n"
            f"{pct:.1f}% windows flagged by PTP",
            fontsize=10,
        )
        ax.set_xlabel("Time (s from recording start)")
        ax.set_ylabel("P(HIGH MWL)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=150)
print(f"\nFigure saved: {OUT_FIG}")
