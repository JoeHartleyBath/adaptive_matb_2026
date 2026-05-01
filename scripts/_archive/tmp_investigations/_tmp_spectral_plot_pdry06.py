"""Spectral plot — PDRY06 control vs adaptation EEG.

Computes Welch PSD per region and for the global average, then plots:
  Panel 1: Global average PSD (all 128 ch), linear-frequency log-power
  Panel 2: Per-region PSD for the 5 model-relevant regions
  Panel 3: Band-power bar chart (delta/theta/alpha/beta/gamma per condition)

Also includes calibration (C1+C2) as reference for the training distribution.

Saves: results/figures/spectral_pdry06.png

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_spectral_plot_pdry06.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import scipy.signal
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams  # noqa: E402
from eeg import EegPreprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SRATE  = 500.0   # load raw (pre-decimate) for cleaner high-freq PSD
TARGET = 128.0
PHYSIO = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio")

FILES = {
    "cal_c1":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf",
    "cal_c2":     PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf",
    "control":    PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf",
    "adaptation": PHYSIO / "sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf",
}
COLORS = {
    "cal_c1":     "#888888",
    "cal_c2":     "#aaaaaa",
    "control":    "#c0392b",
    "adaptation": "#2980b9",
}
LABELS = {
    "cal_c1":     "Calibration C1",
    "cal_c2":     "Calibration C2",
    "control":    "Control",
    "adaptation": "Adaptation",
}
LINESTYLES = {
    "cal_c1":     "--",
    "cal_c2":     ":",
    "control":    "-",
    "adaptation": "-",
}

META       = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES   = META["channel_names"]
FEAT_YAML  = _REPO / "config" / "eeg_feature_extraction.yaml"
REGION_DEF = yaml.safe_load(FEAT_YAML.read_text())["regions"]

# Regions to plot (match feature names used in model)
PLOT_REGIONS = {
    "Frontal Midline": REGION_DEF["FrontalMidline"],
    "Overall Frontal": REGION_DEF["OverallFrontal"],
    "Central":         REGION_DEF["Central"],
    "Parietal":        REGION_DEF["Parietal"],
    "Occipital":       REGION_DEF["Occipital"],
}

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

OUT_FIG = _REPO / "results" / "figures" / "spectral_filtered_pdry06.png"

# ---------------------------------------------------------------------------
# Load EEG
# ---------------------------------------------------------------------------

def load_raw_eeg(path: Path) -> tuple[np.ndarray, float]:
    """Load XDF, decimate to TARGET srate, apply EegPreprocessor. Returns (C×N, srate)."""
    streams, _ = pyxdf.load_xdf(str(path))
    eeg = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T  # C×N
    ts   = np.array(eeg["time_stamps"])
    actual_sr = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else TARGET
    # Decimate to TARGET srate
    if actual_sr > TARGET * 1.1:
        fac  = int(round(actual_sr / TARGET))
        data = data[:, ::fac]
        actual_sr = TARGET
    # Apply pipeline preprocessing (bandpass 0.5–40 Hz + notch 50 Hz)
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    data = pp.process(data)
    return data, actual_sr


def get_channel_indices(ch_list: list[str]) -> list[int]:
    idx = [CH_NAMES.index(c) for c in ch_list if c in CH_NAMES]
    return idx


# ---------------------------------------------------------------------------
# Compute Welch PSDs
# ---------------------------------------------------------------------------

psds: dict[str, tuple[np.ndarray, np.ndarray]] = {}   # condition → (freqs, psd: C×F)

for cond, path in FILES.items():
    print(f"Loading {cond} ...", flush=True)
    data, sr = load_raw_eeg(path)
    n_ch = data.shape[0]
    nperseg = int(sr * 2)   # 2-second segments
    noverlap = nperseg // 2
    freqs, pxx = scipy.signal.welch(
        data, fs=sr, nperseg=nperseg, noverlap=noverlap,
        window="hann", axis=1,
    )
    psds[cond] = (freqs, pxx)
    print(f"  {data.shape[1]} samp, sr={sr:.0f} Hz, {n_ch} ch, PSD shape {pxx.shape}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
freq_mask = freqs <= 50.0   # display up to 50 Hz

n_regions = len(PLOT_REGIONS)
# Layout: 3 rows × 3 cols = 9 panels
# [global, frontal_midline, overall_frontal]
# [central, parietal, occipital]
# [band_bar, <empty>, <empty>]
fig, axes = plt.subplots(3, 3, figsize=(22, 16), constrained_layout=True)
fig.suptitle("PDRY06 — EEG Power Spectral Density (filtered: 0.5–40 Hz bandpass + 50 Hz notch)\n"
             "Control vs Adaptation (calibration = training reference)", fontsize=13)

axes_flat = axes.flatten()

# ── Panel 0: Global average (all channels) ──
ax0 = axes_flat[0]
for cond, (freqs, pxx) in psds.items():
    global_psd = pxx.mean(axis=0)   # mean over channels
    ax0.semilogy(freqs[freq_mask], global_psd[freq_mask],
                 color=COLORS[cond], lw=1.8, ls=LINESTYLES[cond],
                 label=LABELS[cond])
for band, (flo, fhi) in BANDS.items():
    ax0.axvspan(flo, fhi, alpha=0.04, color="grey")
    ax0.text((flo + fhi) / 2, ax0.get_ylim()[0] if ax0.get_ylim()[0] > 0 else 1e-4,
             band, ha="center", va="bottom", fontsize=7, color="grey")
ax0.set_xlabel("Frequency (Hz)")
ax0.set_ylabel("Power (µV²/Hz, log)")
ax0.set_title("Global average (128 ch)")
ax0.legend(fontsize=8)
ax0.set_xlim(1, 50)
ax0.grid(True, which="both", alpha=0.3)

# ── Panels 1–5: Per-region ──
for ax_idx, (region_name, ch_list) in enumerate(PLOT_REGIONS.items(), start=1):
    ax = axes_flat[ax_idx]
    ch_idx = get_channel_indices(ch_list)
    if not ch_idx:
        ax.set_visible(False)
        continue
    for cond, (freqs, pxx) in psds.items():
        region_psd = pxx[ch_idx].mean(axis=0)
        ax.semilogy(freqs[freq_mask], region_psd[freq_mask],
                    color=COLORS[cond], lw=1.5, ls=LINESTYLES[cond],
                    label=LABELS[cond])
    for band, (flo, fhi) in BANDS.items():
        ax.axvspan(flo, fhi, alpha=0.04, color="grey")
    ax.set_title(f"{region_name} (n={len(ch_idx)} ch)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV²/Hz, log)")
    ax.set_xlim(1, 50)
    ax.grid(True, which="both", alpha=0.3)
    if ax_idx == 1:
        ax.legend(fontsize=7)

# ── Panel 6: Band-power bar chart ──
ax_bar = axes_flat[6]
band_names = list(BANDS.keys())
conditions = list(FILES.keys())
n_bands = len(band_names)
n_cond  = len(conditions)
x = np.arange(n_bands)
width = 0.18

def band_power(pxx, freqs, flo, fhi):
    mask = (freqs >= flo) & (freqs <= fhi)
    return float(pxx.mean(axis=0)[mask].sum())  # global mean, integrate

for ci, cond in enumerate(conditions):
    freqs_c, pxx_c = psds[cond]
    vals = [band_power(pxx_c, freqs_c, *BANDS[b]) for b in band_names]
    ax_bar.bar(x + ci * width, vals, width, label=LABELS[cond],
               color=COLORS[cond], alpha=0.85, edgecolor="k", lw=0.4)

ax_bar.set_yscale("log")
ax_bar.set_xticks(x + width * (n_cond - 1) / 2)
ax_bar.set_xticklabels(band_names)
ax_bar.set_ylabel("Band power (µV²/Hz, log)")
ax_bar.set_title("Band power by condition (global avg)")
ax_bar.legend(fontsize=8)
ax_bar.grid(True, axis="y", alpha=0.3)

# Hide unused axes
for ax in axes_flat[7:]:
    ax.set_visible(False)

OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")
