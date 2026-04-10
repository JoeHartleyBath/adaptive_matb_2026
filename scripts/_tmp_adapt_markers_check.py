"""Quick check: adaptation XDF marker timestamps and EEG range."""
import pyxdf, re
from pathlib import Path

XDF = Path(r'C:\data\adaptive_matb\physiology\sub-PSELF\ses-S006\physio\sub-PSELF_ses-S006_task-matb_acq-adaptation_physio.xdf')
streams, _ = pyxdf.load_xdf(str(XDF))
m = next((s for s in streams if 'MATB' in str(s['info']['name'])), None)
evs = [(float(ts), str(x[0]).split('|')[0]) for ts, x in zip(m['time_stamps'], m['time_series'])]
for ts, ev in evs:
    print(f'{ts:.3f}  {ev}')

eeg = next(s for s in streams if int(s['info']['channel_count'][0]) > 4)
eeg_ts = eeg['time_stamps']
print(f'\nEEG start={eeg_ts[0]:.3f}  end={eeg_ts[-1]:.3f}  dur={eeg_ts[-1]-eeg_ts[0]:.1f}s')

# Block durations from START/END pairs
RE = re.compile(r'STUDY/V0/adaptive_automation/\d+/block_(\d+)/(\w+)/(START|END)')
ts_map = {}
for ts, ev in evs:
    mm = RE.match(ev)
    if mm:
        key = f'block_{mm.group(1)}_{mm.group(2)}'
        if mm.group(3) == 'START':
            ts_map[key] = ts
        else:
            start = ts_map.get(key)
            dur = f'{ts - start:.1f}s' if start else 'no-START'
            print(f'  {key}: dur={dur}')
