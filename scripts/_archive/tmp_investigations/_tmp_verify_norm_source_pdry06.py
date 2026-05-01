"""Determine whether PDRY06 norm_stats.json came from the rest XDF or LOW-block fallback.

Recomputes what norm_stats would look like using the rest XDF (the correct path)
and compares against what is on disk. If they are very different, the rest XDF
was NOT used (fallback triggered), which is the root cause of p_high saturation.

Run:
    .\.venv\Scripts\python.exe scripts\_tmp_verify_norm_source_pdry06.py
"""
import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_mwl_training_dataset import (
    PREPROCESSING_CONFIG,
    _merge_eeg_streams,
)
from eeg import EegPreprocessor, extract_windows
from eeg.online_features import OnlineFeatureExtractor

import yaml
import pyxdf
import numpy as np

SRATE    = 128.0
WINDOW_S = 2.0
STEP_S   = 0.25

# Paths
REST_XDF   = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio"
                  r"\sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf")
MODEL_DIR  = Path(r"C:\data\adaptive_matb\models\PDRY06")
FEAT_CFG   = Path(r"C:\adaptive_matb_2026\config\eeg_feature_extraction.yaml")
META_PATH  = Path(r"C:\adaptive_matb_2026\config\eeg_metadata.yaml")

meta     = yaml.safe_load(open(META_PATH))
ch_names = meta["channel_names"]

# Load on-disk norm_stats
with open(MODEL_DIR / "norm_stats.json") as f:
    ns = json.load(f)
disk_mean = np.array(ns["mean"])
disk_std  = np.array(ns["std"])
print(f"On-disk norm_stats (mean): min={disk_mean.min():.4f}  max={disk_mean.max():.4f}  "
      f"median={np.median(disk_mean):.4f}")

# ---------------------------------------------------------------------------
# Recompute from rest XDF using the same path as calibrate_participant.py
# ---------------------------------------------------------------------------
print(f"\nLoading rest XDF: {REST_XDF.name}")
streams, _ = pyxdf.load_xdf(str(REST_XDF))
eeg_stream = _merge_eeg_streams(streams)

if eeg_stream is None:
    print("ERROR: no EEG stream in rest XDF")
    sys.exit(1)

eeg_data = np.array(eeg_stream["time_series"], dtype=np.float32).T  # (n_ch, n)
eeg_ts   = np.array(eeg_stream["time_stamps"])

# Decimate if needed
if len(eeg_ts) > 1:
    actual_srate = (len(eeg_ts) - 1) / (eeg_ts[-1] - eeg_ts[0])
    if actual_srate > SRATE * 1.1:
        factor = int(round(actual_srate / SRATE))
        eeg_data = eeg_data[:, ::factor]
        eeg_ts   = eeg_ts[::factor]
        print(f"  Decimated {actual_srate:.0f} → {SRATE:.0f} Hz (factor={factor})")

print(f"  EEG shape: {eeg_data.shape}  duration: {(eeg_ts[-1]-eeg_ts[0]):.1f}s")

# Look for rest markers (same logic as calibrate_participant.py)
_REST_START = "STUDY/V0/rest/START"
_REST_END   = "STUDY/V0/rest/END"
marker_stream = next(
    (s for s in streams if s["info"]["type"][0] == "Markers"
     and s["info"]["name"][0] == "OpenMATB"), None
)
start_ts = end_ts = None
if marker_stream:
    for ts, sample in zip(marker_stream["time_stamps"], marker_stream["time_series"]):
        label = str(sample[0]).split("|")[0]
        if label == _REST_START:
            start_ts = ts
        elif label == _REST_END:
            end_ts = ts

settle_s = 5.0
if start_ts is None or end_ts is None:
    print("  No REST markers — using full XDF (minus settle)")
    start_ts = eeg_ts[0]
    end_ts   = eeg_ts[-1]
else:
    print(f"  REST markers found: duration={(end_ts-start_ts):.1f}s")

analysis_start = start_ts + settle_s
start_idx = int(np.searchsorted(eeg_ts, analysis_start))
end_idx   = int(np.searchsorted(eeg_ts, end_ts))
print(f"  Analysis window: {start_idx}→{end_idx} samples  ({(end_idx-start_idx)/SRATE:.1f}s)")

# Filter and window
preprocessor = EegPreprocessor(PREPROCESSING_CONFIG)
preprocessor.initialize_filters(eeg_data.shape[0])
filtered = preprocessor.process(eeg_data)

step  = int(STEP_S * SRATE)
win   = int(WINDOW_S * SRATE)
block = filtered[:, start_idx:end_idx]

extractor = OnlineFeatureExtractor(ch_names, srate=SRATE, region_cfg=FEAT_CFG)
feat_rows = []
for s in range(0, block.shape[1] - win, step):
    w = block[:, s : s + win]
    feat_rows.append(extractor.compute(w))

if not feat_rows:
    print("ERROR: no windows extracted from rest XDF")
    sys.exit(1)

rest_X  = np.array(feat_rows)
rest_mean = rest_X.mean(axis=0)
rest_std  = rest_X.std(axis=0)
rest_std[rest_std < 1e-12] = 1.0

print(f"\nRest-derived norm (mean):  min={rest_mean.min():.4f}  max={rest_mean.max():.4f}  "
      f"median={np.median(rest_mean):.4f}")
print(f"  windows={len(feat_rows)}")

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
mean_diff = rest_mean - disk_mean
std_diff  = rest_std  - disk_std
print(f"\nDifference (rest − disk):")
print(f"  mean diff: min={mean_diff.min():.4f}  max={mean_diff.max():.4f}  "
      f"RMS={np.sqrt(np.mean(mean_diff**2)):.4f}")
print(f"  std  diff: min={std_diff.min():.4f}  max={std_diff.max():.4f}  "
      f"RMS={np.sqrt(np.mean(std_diff**2)):.4f}")

# Pearson correlation of mean vectors
corr_mean = float(np.corrcoef(rest_mean, disk_mean)[0, 1])
corr_std  = float(np.corrcoef(rest_std,  disk_std )[0, 1])
print(f"  Pearson r (mean vectors): {corr_mean:.6f}")
print(f"  Pearson r (std  vectors): {corr_std:.6f}")

print()
if corr_mean > 0.999 and np.sqrt(np.mean(mean_diff**2)) < 0.01:
    print("CONCLUSION: norm_stats on disk IS from the rest XDF (matches very closely).")
    print("  The rest XDF was passed to calibrate_participant.py — norm source is correct.")
    print("  p_high saturation has a DIFFERENT cause.")
elif corr_mean < 0.95 or np.sqrt(np.mean(mean_diff**2)) > 0.1:
    print("CONCLUSION: norm_stats on disk does NOT match the rest XDF.")
    print("  calibrate_participant.py likely used the LOW-block fallback.")
    print("  This is the root cause: wrong normalisation baseline → p_high saturation.")
else:
    print("CONCLUSION: ambiguous — partial overlap. Manual inspection needed.")
    print("  Correlation is moderate; may indicate partial rest data was used.")
