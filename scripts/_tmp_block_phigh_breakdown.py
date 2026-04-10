"""Break down MWL stream p_high by scenario block for the S005 adaptation session.

Shows what the model actually output (from the LSL MWL stream recorded in the XDF)
per block, with EMA simulation to check when smoother would cross threshold.
"""
import sys, pyxdf, re, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

xdf_path = Path(
    r"C:\data\adaptive_matb\physiology\sub-PSELF\ses-S005\physio"
    r"\sub-PSELF_ses-S005_task-matb_acq-adaptation_physio.xdf"
)
streams, _ = pyxdf.load_xdf(str(xdf_path))

# MWL stream: [p_high, confidence, quality]
mwl     = next(s for s in streams if s["info"]["type"][0] == "MWL")
mwl_ts  = np.array(mwl["time_stamps"])
mwl_ph  = np.array(mwl["time_series"])[:, 0]
mwl_q   = np.array(mwl["time_series"])[:, 2]

# Block markers
markers = next(s for s in streams if s["info"]["name"][0] == "OpenMATB")
events  = list(zip(markers["time_stamps"], [s[0] for s in markers["time_series"]]))

# AdaptationEvents
adapt_s = next((s for s in streams if s["info"]["name"][0] == "AdaptationEvents"), None)
adapt_events = (
    list(zip(adapt_s["time_stamps"], [s[0] for s in adapt_s["time_series"]]))
    if adapt_s else []
)

# Parse blocks
blocks = []
starts = {}
for ts, ev in events:
    m = re.search(r"block_(\d+)/(\w+)/(START|END)", ev)
    if not m:
        continue
    idx, level, which = m.groups()
    if which == "START":
        starts[idx] = (ts, level)
    elif which == "END":
        s_ts, lev = starts.get(idx, (mwl_ts[0], level))
        blocks.append((idx, lev, s_ts, ts))

# Offline EMA replication (initialise to first sample, same as EmaSmoother)
ALPHA      = 0.05
THRESHOLD  = 0.3248
HYSTERESIS = 0.02

ema = None
smoothed_all = []
for v in mwl_ph:
    if ema is None:
        ema = float(v)
    else:
        ema = ALPHA * float(v) + (1.0 - ALPHA) * ema
    smoothed_all.append(ema)
smoothed_all = np.array(smoothed_all)

t0 = mwl_ts[0]

print(f"Session MWL stream t0 = {t0:.2f}  (all times relative to t0)")
print(f"Threshold = {THRESHOLD:.4f}   Hysteresis = {HYSTERESIS:.4f}")
print()

print("Adaptation events (assist_on / assist_off):")
for ts, ev in adapt_events:
    if ev:
        print(f"  t+{ts - t0:6.1f}s  {ev!r}")
print()

hdr = (
    f"  {'block':<8} {'level':<10} {'t_start':>8} {'t_end':>8} {'dur':>5}"
    f"  {'n':>5}  {'mean_ph':>8}  {'med_ph':>8}"
    f"  {'raw>thr':>8}  {'ema>thr+h':>10}"
)
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for idx, level, t_s, t_e in blocks:
    mask = (mwl_ts >= t_s) & (mwl_ts < t_e)
    ph   = mwl_ph[mask]
    sm   = smoothed_all[mask]
    if len(ph) == 0:
        continue
    pct_raw = (ph > THRESHOLD).mean() * 100
    pct_sm  = (sm > THRESHOLD + HYSTERESIS).mean() * 100
    print(
        f"  block_{idx}  {level:<10} {t_s - t0:8.1f}  {t_e - t0:8.1f}  {t_e - t_s:5.0f}s"
        f"  {len(ph):5d}  {ph.mean():8.3f}  {np.median(ph):8.3f}"
        f"  {pct_raw:7.1f}%  {pct_sm:9.1f}%"
    )

print()
print("First 40 MWL samples (covers warmup + early LOW block):")
print(f"  {'t+s':>6}  {'p_high':>8}  {'quality':>7}  {'EMA':>8}  {'ema>thr+h':>10}")
for i in range(min(40, len(mwl_ts))):
    flag = " <-- assist_on" if abs(mwl_ts[i] - 693574.23) < 0.5 else ""
    print(
        f"  {mwl_ts[i] - t0:6.1f}  {mwl_ph[i]:8.4f}  {mwl_q[i]:7.2f}"
        f"  {smoothed_all[i]:8.4f}  {'YES' if smoothed_all[i] > THRESHOLD + HYSTERESIS else 'no':>10}"
        f"{flag}"
    )
