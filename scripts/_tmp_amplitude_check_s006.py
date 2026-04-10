"""Check raw EEG RMS amplitude across all S006 recordings to quantify signal level change."""
import pyxdf
import numpy as np
from pathlib import Path

PHYSIO = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")

files = {
    "rest_old1":    "sub-PSELF_ses-S006_task-matb_acq-rest_physio_old1.xdf",
    "rest":         "sub-PSELF_ses-S006_task-matb_acq-rest_physio.xdf",
    "cal_c1_old1":  "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio_old1.xdf",
    "cal_c1":       "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf",
    "cal_c2_old1":  "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio_old1.xdf",
    "cal_c2":       "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf",
}

print(f"{'Label':<15} {'Duration':>9} {'srate':>7} {'nchan':>6} {'RMS_raw':>10}  {'MATB_markers':>13}  {'t_start_LSL':>14}")
print("-" * 90)

t_starts = {}
for label, fname in files.items():
    p = PHYSIO / fname
    try:
        streams, _ = pyxdf.load_xdf(str(p))
        eeg = next((s for s in streams
                    if "eego_laptop" in s["info"]["name"][0]
                    and "TRG" not in s["info"]["name"][0]), None)
        if eeg is None:
            print(f"{label:<15}  no EEG stream found")
            continue
        ts   = np.array(eeg["time_stamps"])
        data = np.array(eeg["time_series"], dtype=np.float32)  # (n_samples, n_chan)
        dur  = ts[-1] - ts[0]
        sr   = (len(ts) - 1) / dur if dur > 0 else 0
        rms  = float(np.sqrt(np.mean(data ** 2)))
        nch  = data.shape[1]
        mk   = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)
        n_mk = len(mk["time_stamps"]) if mk else 0
        t0   = ts[0]
        t_starts[label] = t0
        print(f"{label:<15} {dur:9.1f}s {sr:7.0f} {nch:6d} {rms:10.2f}  {n_mk:13d}  {t0:14.3f}")
    except Exception as exc:
        print(f"{label:<15}  ERROR: {exc}")

# Timeline
print()
print("Timeline (LSL seconds):")
prev_label, prev_t = None, None
for label, t in sorted(t_starts.items(), key=lambda kv: kv[1]):
    if prev_t is not None:
        gap = t - prev_t
        print(f"  {prev_label} -> {label:20s}  gap = {gap:.1f}s  ({gap/60:.1f} min)")
    prev_label, prev_t = label, t
