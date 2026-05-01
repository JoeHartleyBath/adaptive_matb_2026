"""Quick diagnostic: per-channel PTP distribution across 2s windows, post-filtering."""
import sys, numpy as np, pyxdf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from build_mwl_training_dataset import PREPROCESSING_CONFIG, _merge_eeg_streams
from eeg import EegPreprocessor

SRATE = 128.0; WIN = 256; STEP = 32

def load_filtered(path):
    streams, _ = pyxdf.load_xdf(str(path))
    eeg = _merge_eeg_streams(streams)
    data = np.array(eeg["time_series"], dtype=np.float32).T
    ts   = np.array(eeg["time_stamps"])
    if len(ts) > 1:
        actual = (len(ts)-1)/(ts[-1]-ts[0])
        if actual > SRATE*1.1:
            fac = int(round(actual/SRATE)); data = data[:,::fac]
    pp = EegPreprocessor(PREPROCESSING_CONFIG)
    pp.initialize_filters(data.shape[0])
    return pp.process(data).astype(np.float64)

files = [
    ("PDRY06-control",     Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio\sub-PDRY06_ses-S001_task-matb_acq-control_physio.xdf")),
    ("PDRY06-adaptation",  Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio\sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf")),
    ("PSELF-S005-control", Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio\sub-PSELF_ses-S005_task-matb_acq-control_physio.xdf")),
    ("PSELF-S005-adapt",   Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio\sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf")),
]

hdr = f"  {'label':<24} {'P1':>9} {'P25':>9} {'P50':>9} {'P75':>9} {'P99':>9}  {'max':>9}"
print(hdr + "  (per-channel PTP, 2s window, Volts)")
print("  " + "-" * 75)
for label, path in files:
    data = load_filtered(path)
    n = data.shape[1]
    all_ptp = []
    for s in range(WIN, n - STEP, STEP):
        w = data[:, s-WIN : s]
        all_ptp.append(np.ptp(w, axis=1))
    A = np.concatenate(all_ptp)
    ps = np.percentile(A, [1, 25, 50, 75, 99])
    print(f"  {label:<24} {ps[0]:9.2e} {ps[1]:9.2e} {ps[2]:9.2e} {ps[3]:9.2e} {ps[4]:9.2e}  {A.max():9.2e}")

print()
print("  Reference scale:")
print("    100 µV = 1.0e-4 V  (MNE standard epoch rejection, post-ICA)")
print("    200 µV = 2.0e-4 V  (previous PTP_THRESHOLD_V)")
print("    1 mV   = 1.0e-3 V")
print("    5 mV   = 5.0e-3 V")
