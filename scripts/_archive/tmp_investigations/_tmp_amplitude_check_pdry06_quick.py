"""Quick RMS amplitude check across PDRY06 conditions."""
import sys, numpy as np, pyxdf
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from build_mwl_training_dataset import _merge_eeg_streams

PHYSIO = r'C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio'
SRATE = 128.0
files = [
    ('rest',       'sub-PDRY06_ses-S001_task-matb_acq-rest_physio.xdf'),
    ('cal_c1',     'sub-PDRY06_ses-S001_task-matb_acq-cal_c1_physio.xdf'),
    ('cal_c2',     'sub-PDRY06_ses-S001_task-matb_acq-cal_c2_physio.xdf'),
    ('control',    'sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf'),
    ('adaptation', 'sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf'),
]

print(f"  {'condition':<12}  {'global_rms':>12}  {'median_ch':>10}  {'p1_ch':>9}  {'p99_ch':>9}  {'n_samp':>8}")
print("  " + "-" * 72)
refs = {}
for label, fname in files:
    streams, _ = pyxdf.load_xdf(f'{PHYSIO}/{fname}')
    eeg = _merge_eeg_streams(streams)
    data = np.array(eeg['time_series'], dtype=np.float32).T  # (C, N)
    ts   = np.array(eeg['time_stamps'])
    if len(ts) > 1:
        actual = (len(ts) - 1) / (ts[-1] - ts[0])
        if actual > SRATE * 1.1:
            fac = int(round(actual / SRATE)); data = data[:, ::fac]
    global_rms  = float(np.sqrt(np.mean(data ** 2)))
    per_ch_rms  = np.sqrt(np.mean(data ** 2, axis=1))
    refs[label] = global_rms
    print(f"  {label:<12}  {global_rms:12.6f}  {np.median(per_ch_rms):10.6f}  "
          f"{np.percentile(per_ch_rms, 1):9.6f}  {np.percentile(per_ch_rms, 99):9.6f}  "
          f"{data.shape[1]:8d}")

cal_rms = (refs['cal_c1'] + refs['cal_c2']) / 2
print()
print("  Ratio vs calibration mean:")
for k, v in refs.items():
    print(f"    {k:<12}  {v / cal_rms:.3f}x  ({v:.6f})")
