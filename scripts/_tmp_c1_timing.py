"""Quick timing check: C1 EEG start vs C2 to quantify scenario fallback offset error."""
from pathlib import Path
import pyxdf
import numpy as np

PHYSIO = Path(r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio")

streams1, _ = pyxdf.load_xdf(str(PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c1_physio.xdf"))
eeg1 = next(s for s in streams1 if "eego_laptop" in s["info"]["name"][0] and "TRG" not in s["info"]["name"][0])
ts1 = np.array(eeg1["time_stamps"])
print(f"C1 EEG  t_start={ts1[0]:.3f}  t_end={ts1[-1]:.3f}  duration={ts1[-1]-ts1[0]:.1f}s")

streams2, _ = pyxdf.load_xdf(str(PHYSIO / "sub-PSELF_ses-S006_task-matb_acq-cal_c2_physio.xdf"))
eeg2 = next(s for s in streams2 if "eego_laptop" in s["info"]["name"][0] and "TRG" not in s["info"]["name"][0])
mk2  = next(s for s in streams2 if s["info"]["name"][0] == "OpenMATB")
ts2 = np.array(eeg2["time_stamps"])
actual_offset = float(mk2["time_stamps"][0]) - float(ts2[0])
print(f"C2 EEG  t_start={ts2[0]:.3f}  duration={ts2[-1]-ts2[0]:.1f}s")
print(f"C2 first MATB marker offset: {actual_offset:.3f}s  ({mk2['time_series'][0][0][:60]})")

gap = float(ts2[0]) - float(ts1[-1])
print(f"\nC1 end -> C2 start gap: {gap:.1f}s  ({gap/60:.1f} min)")

assumed = 12.0
err = assumed - actual_offset
print(f"\nAssumed fallback offset: {assumed}s")
print(f"Actual offset (from C2):  {actual_offset:.2f}s")
print(f"Offset error:             {err:.2f}s too late")
print(f"Contamination per block:  {err/59*100:.1f}%  of each 59s block from the next block")
