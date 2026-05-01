"""25 Hz artifact sweep — all XDF recordings.

For every non-backup XDF in the physiology data directory, loads the first
60 s of EEG, applies the standard EegPreprocessor (0.5–40 Hz bandpass +
50 Hz notch), and computes a Welch PSD on the global channel mean.  The
"spike ratio" at 25 Hz is defined as:

    spike_ratio = P(25 Hz) / mean(P(22–24 Hz), P(26–28 Hz))

A ratio > 1 means 25 Hz is elevated above its local floor.
A ratio > 3 is flagged as a clear spike (arbitrary but conservative threshold).

Also computes the absolute PSD around 25 Hz so amplitude can be compared
across sessions.

Output:
    results/25hz_sweep_all_xdfs.csv   — per-file table
    results/figures/25hz_sweep_heatmap.png  — heatmap: participant × run

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_25hz_sweep_all_xdfs.py
"""
from __future__ import annotations

import re
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
PHYSIO_ROOT  = Path(r"C:\data\adaptive_matb\physiology")
TARGET_SR    = 128.0
MAX_SECS     = 60        # only load this much per file (for speed)
SPIKE_THRESH = 3.0       # spike_ratio above this = "clear spike" (p90 across channels)

# Note: spike ratio is computed per-channel; we report median, p90, and max.
# p90 >= SPIKE_THRESH means at least 10% of channels have a clear 25 Hz spike.

# Adjacent band for normalisation (Hz, exclusive of 25 Hz bin itself)
FLOOR_BAND = (22.0, 28.0)  # will mask out [24.5, 25.5] Hz as the spike region

OUT_CSV = _REPO / "results" / "25hz_sweep_all_xdfs.csv"
OUT_FIG = _REPO / "results" / "figures" / "25hz_sweep_heatmap.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

META     = yaml.safe_load((_REPO / "config" / "eeg_metadata.yaml").read_text())
CH_NAMES = META["channel_names"]
N_CH_EXPECTED = len(CH_NAMES)

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
all_xdfs = sorted(
    p for p in PHYSIO_ROOT.rglob("*.xdf")
    if "_old" not in p.name
)
print(f"Found {len(all_xdfs)} XDF files (non-backup)\n")

# ---------------------------------------------------------------------------
# Parse participant/session/acq from BIDS-ish filename
# ---------------------------------------------------------------------------
_BIDS_RE = re.compile(
    r"sub-(?P<pid>\w+)_ses-(?P<ses>\w+)_task-\w+_acq-(?P<acq>.+)_physio"
)

def parse_bids(path: Path) -> dict[str, str]:
    m = _BIDS_RE.search(path.stem)
    if m:
        return m.groupdict()
    return {"pid": path.parent.parent.parent.name, "ses": "?", "acq": path.stem}


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
rows: list[dict] = []

for xdf_path in all_xdfs:
    info = parse_bids(xdf_path)
    label = f"{info['pid']}/{info['ses']}/{info['acq']}"
    print(f"  {label}", flush=True)

    try:
        streams, _ = pyxdf.load_xdf(str(xdf_path))
    except Exception as exc:
        print(f"    FAILED to load: {exc}")
        rows.append({**info, "status": "load_error", "duration_s": 0,
                     "spike_ratio": float("nan"), "p25_dB": float("nan"),
                     "p25_raw": float("nan"), "floor_dB": float("nan")})
        continue

    eeg = _merge_eeg_streams(streams)
    if eeg is None:
        print(f"    SKIPPED (no EEG stream)")
        rows.append({**info, "status": "no_eeg", "duration_s": 0,
                     "spike_ratio": float("nan"), "p25_dB": float("nan"),
                     "p25_raw": float("nan"), "floor_dB": float("nan")})
        continue

    n_ch = int(eeg["info"]["channel_count"][0])
    if n_ch != N_CH_EXPECTED:
        print(f"    SKIPPED (channel count {n_ch} != {N_CH_EXPECTED})")
        rows.append({**info, "status": f"ch_mismatch_{n_ch}", "duration_s": 0,
                     "spike_ratio": float("nan"), "p25_dB": float("nan"),
                     "p25_raw": float("nan"), "floor_dB": float("nan")})
        continue

    data = np.array(eeg["time_series"], dtype=np.float32).T   # C×N
    ts   = np.array(eeg["time_stamps"])
    duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    # Decimate to TARGET_SR
    actual_sr = TARGET_SR
    if len(ts) > 1:
        raw_sr = (len(ts) - 1) / (ts[-1] - ts[0])
        if raw_sr > TARGET_SR * 1.1:
            fac  = int(round(raw_sr / TARGET_SR))
            data = data[:, ::fac]
            ts   = ts[::fac]
            actual_sr = TARGET_SR

    # Trim to first MAX_SECS seconds
    max_samp = int(MAX_SECS * actual_sr)
    if data.shape[1] > max_samp:
        data = data[:, :max_samp]

    # Apply standard preprocessing
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    data = pp.process(data)

    # Per-channel PSDs → spike ratio for each channel
    min_seg = int(actual_sr * 2)  # minimum 2 s segment
    if data.shape[1] < min_seg:
        print(f"    SKIPPED (too short: {data.shape[1]} samp)")
        rows.append({**info, "status": "too_short", "duration_s": round(duration_s, 1),
                     "spike_med": float("nan"), "spike_p90": float("nan"),
                     "spike_max": float("nan")})
        continue
    nperseg = min(int(actual_sr * 4), data.shape[1] // 2)
    freqs, pxx = scipy.signal.welch(
        data, fs=actual_sr, nperseg=nperseg, noverlap=nperseg // 2,
        window="hann", axis=1,   # pxx shape: C × F
    )

    # Spike ratio per channel
    i25      = int(np.argmin(np.abs(freqs - 25.0)))
    floor_mask = (
        (freqs >= FLOOR_BAND[0]) & (freqs <= FLOOR_BAND[1])
        & ~((freqs >= 24.5) & (freqs <= 25.5))
    )
    if floor_mask.sum() < 2:
        floor_mask = (
            ((freqs >= 22.0) & (freqs <= 24.0)) |
            ((freqs >= 26.0) & (freqs <= 28.0))
        )
    p25_per_ch   = pxx[:, i25]                              # C,
    p_floor_per_ch = pxx[:, floor_mask].mean(axis=1)        # C,
    valid_ch = p_floor_per_ch > 0
    ratios_per_ch = np.where(valid_ch,
                             p25_per_ch / np.where(valid_ch, p_floor_per_ch, 1.0),
                             np.nan)

    spike_med = float(np.nanmedian(ratios_per_ch))
    spike_p90 = float(np.nanpercentile(ratios_per_ch, 90))
    spike_max = float(np.nanmax(ratios_per_ch))
    # worst channel index
    worst_ch  = int(np.nanargmax(ratios_per_ch))
    worst_name = CH_NAMES[worst_ch] if worst_ch < len(CH_NAMES) else str(worst_ch)

    flag = "SPIKE" if spike_p90 >= SPIKE_THRESH else "ok"
    print(f"    dur={duration_s:.0f}s  "
          f"spike_med={spike_med:.2f}  p90={spike_p90:.2f}  max={spike_max:.2f}  "
          f"worst_ch={worst_name}  [{flag}]")

    rows.append({**info, "status": flag, "duration_s": round(duration_s, 1),
                 "spike_med": round(spike_med, 3),
                 "spike_p90": round(spike_p90, 3),
                 "spike_max": round(spike_max, 3),
                 "worst_ch":  worst_name})

# ---------------------------------------------------------------------------
# Write CSV
# ---------------------------------------------------------------------------
import csv
fieldnames = ["pid", "ses", "acq", "status", "duration_s",
              "spike_med", "spike_p90", "spike_max", "worst_ch"]
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV saved: {OUT_CSV}")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
valid = [r for r in rows if not np.isnan(r["spike_p90"])]
spiked = [r for r in valid if r["spike_p90"] >= SPIKE_THRESH]
print(f"\n{'='*60}")
print(f"SUMMARY  ({len(valid)} files with valid PSD)")
print(f"  spike_p90 >= {SPIKE_THRESH}: {len(spiked)} / {len(valid)} files")
print(f"{'='*60}")
print(f"\n{'Participant':<12} {'Ses':<6} {'Acq':<30} {'spike_med':>9} {'spike_p90':>9} {'spike_max':>9}  worst_ch")
print("-" * 90)
for r in sorted(valid, key=lambda x: -x["spike_p90"]):
    flag = " *** SPIKE ***" if r["spike_p90"] >= SPIKE_THRESH else ""
    print(f"{r['pid']:<12} {r['ses']:<6} {r['acq']:<30} "
          f"{r['spike_med']:>9.2f} {r['spike_p90']:>9.2f} {r['spike_max']:>9.2f}  "
          f"{r.get('worst_ch','?')}{flag}")

# ---------------------------------------------------------------------------
# Heatmap: participants (rows) × acquisition tags (cols), colour = spike_ratio
# ---------------------------------------------------------------------------
pids = sorted({r["pid"] for r in valid})
acqs_all = sorted({r["acq"] for r in valid})

# Build 2D matrix indexed by (pid+ses, acq)
keys = sorted({f"{r['pid']}/{r['ses']}" for r in valid})
mat  = np.full((len(keys), len(acqs_all)), np.nan)
for r in valid:
    ki = keys.index(f"{r['pid']}/{r['ses']}")
    ai = acqs_all.index(r["acq"])
    mat[ki, ai] = r["spike_p90"]  # use p90 for heatmap

fig, ax = plt.subplots(figsize=(max(12, len(acqs_all) * 0.7), max(6, len(keys) * 0.45)),
                        constrained_layout=True)
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
               vmin=0.5, vmax=8.0, interpolation="nearest")
plt.colorbar(im, ax=ax, label="Spike ratio (P(25 Hz) / local floor)")
ax.axhline(y=-0.5, color="gray", lw=0.5)  # Just a placeholder
ax.set_xticks(range(len(acqs_all)))
ax.set_xticklabels(acqs_all, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(keys)))
ax.set_yticklabels(keys, fontsize=8)
ax.set_title(f"25 Hz spike ratio across all XDF recordings (threshold = {SPIKE_THRESH}×)")
ax.axvline(x=-0.5, color="gray", lw=0.5)

# Mark clear spikes
for ki in range(len(keys)):
    for ai in range(len(acqs_all)):
        v = mat[ki, ai]
        if not np.isnan(v) and v >= SPIKE_THRESH:
            ax.text(ai, ki, f"{v:.1f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="white")
        elif not np.isnan(v):
            ax.text(ai, ki, f"{v:.1f}", ha="center", va="center",
                    fontsize=6, color="black")

plt.savefig(OUT_FIG, dpi=150)
print(f"\nHeatmap saved: {OUT_FIG}")
print("DONE.")
