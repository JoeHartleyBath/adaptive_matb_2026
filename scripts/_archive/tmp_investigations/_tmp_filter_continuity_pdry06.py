"""_tmp_filter_continuity_pdry06.py

Tests the IIR filter-state contamination hypothesis for the PDRY06
live vs offline MWL divergence.

The MWL estimator creates a fresh EegPreprocessor when it connects to LSL.
*Before* the adaptation XDF recording starts (LabRecorder is restarted after
the estimator connects), the estimator's filter may have processed a short
burst of LSL-buffered EEG that does not appear in the XDF.

This script simulates three conditions for adaptation-run inference:

  A: Fresh filter, adaptation XDF only                      (true offline baseline)
  B: Full control XDF → adaptation XDF, single preprocessor (worst-case contamination)
  C: Last 30 s of control XDF → adaptation XDF              (approx 2 s LSL warmup + margin)

If filter contamination explains the live p_high ≈ 0.97 we would expect
r(B or C, live_MWL) >> r(A, live_MWL).  If contamination does NOT explain
it, all three offline cases will cluster near p_high ≈ 0.14.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_filter_continuity_pdry06.py
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams  # noqa: E402
from eeg import EegPreprocessor  # noqa: E402
from eeg.online_features import OnlineFeatureExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

PHYSIO    = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")
MODEL_DIR = Path(r"C:\data\adaptive_matb\models\PDRY06")
OUT_FIG   = Path(r"C:\adaptive_matb_2026\results\figures\filter_continuity_pdry06.png")

CTRL_XDF  = PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf"
ADAPT_XDF = PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

meta     = yaml.safe_load(open(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml"))
ch_names: list[str] = meta["channel_names"]
feat_cfg = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")

pipeline = joblib.load(MODEL_DIR / "pipeline.pkl")
selector = joblib.load(MODEL_DIR / "selector.pkl")

with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
with open(MODEL_DIR / "model_config.json") as f:
    model_cfg = json.load(f)

norm_mean = np.array(ns["mean"], dtype=np.float64)
norm_std  = np.array(ns["std"],  dtype=np.float64)
norm_std[norm_std < 1e-12] = 1.0
n_classes = int(ns.get("n_classes", 3))
threshold = float(model_cfg["youden_threshold"])
print(f"Model: {n_classes}-class  threshold={threshold:.4f}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eeg(xdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XDF → decimate to SRATE. Returns (data (C,N), timestamps (N,))."""
    streams, _ = pyxdf.load_xdf(str(xdf_path))
    eeg  = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac  = int(round(actual / SRATE))
            data = data[:, ::fac]
            ts   = ts[::fac]
    return data, ts


def infer_on_filtered(filtered: np.ndarray) -> np.ndarray:
    """Windowed inference on already-filtered EEG. Returns p_high (N,)."""
    ext  = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=feat_cfg)
    step = int(STEP_S   * SRATE)
    win  = int(WINDOW_S * SRATE)
    n    = filtered.shape[1]
    out  = []
    for s in range(win, n - step, step):
        f   = ext.compute(filtered[:, s - win : s])
        fz  = (f - norm_mean) / norm_std
        p   = pipeline.predict_proba(selector.transform(fz[np.newaxis, :]))[0]
        out.append(float(p[-1]))
    return np.array(out)


def filter_with(preprocessor: EegPreprocessor, data: np.ndarray) -> np.ndarray:
    """Run preprocessor.process() and return result (does not re-init)."""
    return preprocessor.process(data)


# ---------------------------------------------------------------------------
# Load EEG
# ---------------------------------------------------------------------------

print("Loading control XDF  ...", flush=True)
ctrl_data, ctrl_ts = load_eeg(CTRL_XDF)
n_ch = ctrl_data.shape[0]
print(f"  {ctrl_data.shape[1]} samp ({ctrl_data.shape[1]/SRATE:.0f} s)  {n_ch} ch")

print("Loading adaptation XDF ...", flush=True)
adapt_data, adapt_ts = load_eeg(ADAPT_XDF)
print(f"  {adapt_data.shape[1]} samp ({adapt_data.shape[1]/SRATE:.0f} s)  {n_ch} ch")

# ---------------------------------------------------------------------------
# Case A: fresh filter, adaptation only  (true offline reference)
# ---------------------------------------------------------------------------
print("\nCase A — fresh filter, adaptation XDF only ...")
pp_A = EegPreprocessor(PREPROCESSING_CONFIG)
pp_A.initialize_filters(n_ch)
filt_A = filter_with(pp_A, adapt_data)
ph_A   = infer_on_filtered(filt_A)
print(f"  n={len(ph_A)}  mean={ph_A.mean():.3f}  pct_above={np.mean(ph_A > threshold)*100:.1f}%")

# ---------------------------------------------------------------------------
# Case B: full control XDF → adaptation (single persistent preprocessor)
# ---------------------------------------------------------------------------
print("\nCase B — full control → adaptation (persistent filter) ...")
pp_B = EegPreprocessor(PREPROCESSING_CONFIG)
pp_B.initialize_filters(n_ch)
filter_with(pp_B, ctrl_data)           # warm on control; output discarded
filt_B = filter_with(pp_B, adapt_data) # continue into adaptation
ph_B   = infer_on_filtered(filt_B)
print(f"  n={len(ph_B)}  mean={ph_B.mean():.3f}  pct_above={np.mean(ph_B > threshold)*100:.1f}%")

# ---------------------------------------------------------------------------
# Case C: last 30 s of control → adaptation (approx. LSL warmup burst)
# ---------------------------------------------------------------------------
PRIOR_S = 30.0
n_prior  = int(PRIOR_S * SRATE)
prior    = ctrl_data[:, -n_prior:] if ctrl_data.shape[1] >= n_prior else ctrl_data
print(f"\nCase C — last {PRIOR_S:.0f}s of control → adaptation ...")
pp_C = EegPreprocessor(PREPROCESSING_CONFIG)
pp_C.initialize_filters(n_ch)
filter_with(pp_C, prior)
filt_C = filter_with(pp_C, adapt_data)
ph_C   = infer_on_filtered(filt_C)
print(f"  n={len(ph_C)}  mean={ph_C.mean():.3f}  pct_above={np.mean(ph_C > threshold)*100:.1f}%")

# ---------------------------------------------------------------------------
# MWL stream from adaptation XDF (the live signal)
# ---------------------------------------------------------------------------
print("\nLoading live MWL stream from adaptation XDF ...")
streams_a, _ = pyxdf.load_xdf(str(ADAPT_XDF))
mwl_s = next((s for s in streams_a if s["info"]["type"][0] == "MWL"), None)
if mwl_s is not None:
    live_ph = np.array(mwl_s["time_series"], dtype=np.float32)[:, 0]
    live_ts = np.array(mwl_s["time_stamps"])
    print(f"  n={len(live_ph)}  mean={live_ph.mean():.3f}  pct_above={np.mean(live_ph > threshold)*100:.1f}%")
else:
    print("  WARNING: No MWL stream found.")
    live_ph = live_ts = None

# ---------------------------------------------------------------------------
# Pearson r between each offline case and the live MWL stream
# ---------------------------------------------------------------------------
win_samp = int(WINDOW_S * SRATE)
step_samp = int(STEP_S  * SRATE)
n_adapt = adapt_data.shape[1]
starts   = list(range(win_samp, n_adapt - step_samp, step_samp))
# Timestamps at end of each window (matches estimator's "most recent 2s" position)
win_ts = adapt_ts[[min(s, len(adapt_ts) - 1) for s in starts]]

if live_ph is not None:
    valid = (win_ts >= live_ts[0]) & (win_ts <= live_ts[-1])
    n_valid = valid.sum()
    if n_valid >= 20:
        live_interp = np.interp(win_ts[valid], live_ts, live_ph)
        r_A = pearsonr(ph_A[:len(valid)][valid], live_interp)[0]
        r_B = pearsonr(ph_B[:len(valid)][valid], live_interp)[0]
        r_C = pearsonr(ph_C[:len(valid)][valid], live_interp)[0]
        print(f"\n  Pearson r vs live MWL stream ({n_valid} matched windows):")
        print(f"    A (fresh):       r = {r_A:.3f}")
        print(f"    B (full ctrl):   r = {r_B:.3f}")
        print(f"    C (last 30s):    r = {r_C:.3f}")
        print()
        if max(r_B, r_C) > r_A + 0.3:
            print("  >> Filter contamination plausible (B or C substantially closer to live)")
        else:
            print("  >> Filter contamination NOT the primary cause (B≈C≈A)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# --- Top panel: time series ---
ax = axes[0]
t_off = win_ts - adapt_ts[0]                  # time relative to XDF start

ax.plot(t_off[:len(ph_A)], ph_A, lw=0.9, color="steelblue",  alpha=0.85,
        label="A: fresh-offline (adapt only)")
ax.plot(t_off[:len(ph_B)], ph_B, lw=0.9, color="firebrick",  alpha=0.85,
        label="B: full-control → adapt")
ax.plot(t_off[:len(ph_C)], ph_C, lw=0.9, color="darkorange", alpha=0.85,
        label=f"C: last {PRIOR_S:.0f}s ctrl → adapt")
if live_ph is not None:
    ax.plot(live_ts - adapt_ts[0], live_ph, lw=1.2, color="green", alpha=0.9,
            label="Live XDF MWL stream")
ax.axhline(threshold, color="k", ls="--", lw=0.8, alpha=0.5,
           label=f"threshold={threshold:.3f}")
ax.set_xlabel("Time from XDF start (s)")
ax.set_ylabel("P(HIGH)")
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=8, loc="upper right")
ax.set_title("PDRY06 — Filter Continuity: adaptation P(HIGH) by preprocessing path")

# --- Bottom panel: bar chart ---
ax2 = axes[1]
labels = [
    f"A\nfresh\noffline",
    f"B\nfull ctrl\n→ adapt",
    f"C\nlast {PRIOR_S:.0f}s\n→ adapt",
    "Live\nXDF MWL",
]
vals = [
    np.mean(ph_A > threshold) * 100,
    np.mean(ph_B > threshold) * 100,
    np.mean(ph_C > threshold) * 100,
    (np.mean(live_ph > threshold) * 100 if live_ph is not None else float("nan")),
]
colors = ["steelblue", "firebrick", "darkorange", "green"]
bars = ax2.bar(labels, vals, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
ax2.set_ylabel("% windows above threshold")
ax2.set_ylim(0, 115)
ax2.set_title("% above Youden threshold by inference path")
for bar, val in zip(bars, vals):
    if not (np.isnan(val) if isinstance(val, float) else False):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"Figure saved: {OUT_FIG}")
