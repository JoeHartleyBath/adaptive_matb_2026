"""Compare the live MWL stream recorded in the adaptation XDF against mwl_audit.csv.

This settles whether the MWL estimator was genuinely outputting p_high≈1.0
during the adaptation run (matching audit) or whether the XDF MWL channel
shows something different (which would point to an audit logging bug).

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_compare_mwl_xdf_vs_audit.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XDF_PATH   = Path(r"C:\data\adaptive_matb\physiology\sub-PDRY06\ses-S001\physio"
                  r"\sub-PDRY06_ses-S001_task-matb_acq-adaptation_physio.xdf")
AUDIT_CSV  = Path(r"C:\data\adaptive_matb\openmatb\PDRY06\S001\mwl_audit.csv")
OUT_FIG    = Path(r"C:\adaptive_matb_2026\results\figures\mwl_xdf_vs_audit_pdry06.png")

# ---------------------------------------------------------------------------
# Load adaptation XDF — find MWL stream
# ---------------------------------------------------------------------------
print(f"Loading XDF: {XDF_PATH.name}")
streams, header = pyxdf.load_xdf(str(XDF_PATH))

mwl_stream = None
for s in streams:
    stype = s["info"]["type"][0] if s["info"].get("type") else ""
    sname = s["info"]["name"][0] if s["info"].get("name") else ""
    n_ch  = int(s["info"]["channel_count"][0])
    n_samp = len(s["time_stamps"])
    print(f"  stream: name={sname!r:20s}  type={stype!r:10s}  ch={n_ch}  samples={n_samp}")
    if stype == "MWL":
        mwl_stream = s

if mwl_stream is None:
    print("\nERROR: No MWL-type stream found in XDF. The MWL outlet was NOT recorded.")
    print("This means LabRecorder restarted before the MWL estimator outlet appeared,")
    print("or the estimator was not running when LabRecorder captured its stream list.")
    sys.exit(1)

mwl_data = np.array(mwl_stream["time_series"])   # (n_samp, 3)  [mwl_value, confidence, quality]
mwl_ts   = np.array(mwl_stream["time_stamps"])   # LSL timestamps

p_high_xdf  = mwl_data[:, 0]
confidence  = mwl_data[:, 1]
quality     = mwl_data[:, 2]

# Relative time from scenario start (first sample)
t_xdf = mwl_ts - mwl_ts[0]

print(f"\nMWL stream in XDF:")
print(f"  samples : {len(p_high_xdf)}")
print(f"  duration: {t_xdf[-1]:.1f} s")
print(f"  p_high  : mean={p_high_xdf.mean():.4f}  std={p_high_xdf.std():.4f}  "
      f"min={p_high_xdf.min():.4f}  max={p_high_xdf.max():.4f}")
print(f"  quality : mean={quality.mean():.3f}")

# ---------------------------------------------------------------------------
# Load mwl_audit.csv
# ---------------------------------------------------------------------------
audit = pd.read_csv(AUDIT_CSV)
p_high_audit = audit["mwl_raw"].values
t_audit      = audit["scenario_time_s"].values

print(f"\nmwl_audit.csv:")
print(f"  rows    : {len(audit)}")
print(f"  duration: {t_audit[-1]:.1f} s")
print(f"  p_high  : mean={p_high_audit.mean():.4f}  std={p_high_audit.std():.4f}  "
      f"min={p_high_audit.min():.4f}  max={p_high_audit.max():.4f}")

# ---------------------------------------------------------------------------
# Threshold comparison
# ---------------------------------------------------------------------------
threshold = float(audit["threshold"].iloc[0])
pct_above_xdf   = (p_high_xdf   > threshold).mean() * 100
pct_above_audit = (p_high_audit > threshold).mean() * 100
print(f"\nThreshold = {threshold:.4f}")
print(f"  XDF  : {pct_above_xdf:.1f}% above threshold")
print(f"  Audit: {pct_above_audit:.1f}% above threshold")

# ---------------------------------------------------------------------------
# Agreement check
# ---------------------------------------------------------------------------
# Align by nearest timestamp (XDF ts vs audit LSL timestamp)
audit_ts = audit["timestamp_lsl"].values
# For each audit row find closest XDF sample
from scipy.interpolate import interp1d

if len(mwl_ts) > 1:
    interp_fn = interp1d(mwl_ts, p_high_xdf, bounds_error=False, fill_value=np.nan)
    p_high_xdf_aligned = interp_fn(audit_ts)
    valid = ~np.isnan(p_high_xdf_aligned)
    if valid.sum() > 0:
        diff = p_high_xdf_aligned[valid] - p_high_audit[valid]
        print(f"\nPoint-to-point alignment (XDF interpolated @ audit timestamps, n={valid.sum()}):")
        print(f"  mean |diff| = {np.abs(diff).mean():.6f}")
        print(f"  max  |diff| = {np.abs(diff).max():.6f}")
        if np.abs(diff).max() < 0.001:
            print("  -> XDF and audit are IDENTICAL (same estimator output, just recorded twice).")
        elif np.abs(diff).mean() < 0.02:
            print("  -> XDF and audit are very close (minor timestamp jitter only).")
        else:
            print("  -> XDF and audit DIVERGE — estimator output changed between XDF and audit recording.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

ax = axes[0]
ax.plot(t_audit, p_high_audit, color="steelblue", lw=0.8, label="mwl_audit.csv (mwl_raw)")
ax.plot(t_xdf,   p_high_xdf,   color="orange",    lw=0.8, alpha=0.8, label="XDF MWL stream")
ax.axhline(threshold, color="red", lw=1.2, ls="--", label=f"threshold={threshold:.4f}")
ax.set_xlabel("Scenario time (s)")
ax.set_ylabel("p_high")
ax.set_title("PDRY06 Adaptation — MWL: XDF stream vs audit CSV")
ax.legend(fontsize=8)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Histogram comparison
ax2 = axes[1]
ax2.hist(p_high_audit, bins=50, alpha=0.5, color="steelblue", label="audit CSV", density=True)
ax2.hist(p_high_xdf,   bins=50, alpha=0.5, color="orange",    label="XDF stream", density=True)
ax2.axvline(threshold, color="red", lw=1.2, ls="--", label=f"threshold={threshold:.4f}")
ax2.set_xlabel("p_high")
ax2.set_ylabel("Density")
ax2.set_title("Distribution comparison")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_FIG, dpi=150)
print(f"\nFigure saved: {OUT_FIG}")
