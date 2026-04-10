"""Fast figure regeneration for the +block_01 model MWL adaptation timeline.

Loads pre-computed arrays from the cache written by _tmp_retrain_with_block01.py.
Edit the plotting section freely and re-run this script in seconds.

Run the main script first (once) to populate the cache:
    python scripts/_tmp_retrain_with_block01.py

Then iterate here:
    python scripts/_tmp_plot_new_model_fig.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pyxdf

CACHE_DIR = Path(r"C:\adaptive_matb_2026\results\_tmp_new_model_cache")
OUT_PATH  = Path(r"C:\adaptive_matb_2026\results\figures") / \
            "adaptation_pself__fig01__mwl_timeline_new_model.png"

BLOCK_DURATION_S = 59   # adaptation scenario: blocks are 0:00:XX → 0:XX:59

_LEVEL_COLOURS = {"HIGH": "tab:red", "MODERATE": "tab:orange", "LOW": "tab:blue"}

# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------

cache_npz  = CACHE_DIR / "adapt_inference.npz"
cache_meta = CACHE_DIR / "meta.json"

if not cache_npz.exists() or not cache_meta.exists():
    sys.exit(
        "ERROR: cache not found.\n"
        "Run 'python scripts/_tmp_retrain_with_block01.py' first to generate it."
    )

arrays = np.load(cache_npz)
ph_new   = arrays["ph_new"]
ema_new  = arrays["ema_new"]
on_new   = arrays["on_new"].astype(bool)
t_rel    = arrays["t_rel"]
lsl_ts   = arrays["lsl_ts"]

meta = json.load(open(cache_meta))
thr_full    = float(meta["threshold"])
adapt_blocks = [(s, e, lv) for s, e, lv in meta["adapt_blocks"]]   # recorded blocks only
adapt_xdf   = Path(meta["adapt_xdf"])
t0          = float(meta["t0"])

print(f"Loaded cache: {cache_npz.name} + meta.json")
print(f"  {len(ph_new)} windows, threshold={thr_full:.4f}")


# ---------------------------------------------------------------------------
# Recover block_01 from the adaptation XDF (START at t=0 → missed by LabRecorder)
# ---------------------------------------------------------------------------

BLOCK_RE = re.compile(r"block_01/(?P<level>LOW|MODERATE|HIGH)/END")

streams, _ = pyxdf.load_xdf(str(adapt_xdf))
matb_stream = next((s for s in streams if s["info"]["name"][0] == "OpenMATB"), None)

recovered_block01 = None
if matb_stream is not None:
    for ts, row in zip(matb_stream["time_stamps"], matb_stream["time_series"]):
        ev = row[0]
        m = BLOCK_RE.search(ev)
        if m:
            b01_end_lsl = float(ts)
            b01_end_rel = b01_end_lsl - t0
            b01_start_rel = b01_end_rel - BLOCK_DURATION_S
            recovered_block01 = (b01_start_rel, b01_end_rel, m.group("level"))
            print(f"  Recovered block_01/{m.group('level')}: "
                  f"t={b01_start_rel:.1f}s → {b01_end_rel:.1f}s")
            break

# Build full block list: recovered block_01 first, then the recorded ones
all_blocks_rel: list[tuple[float, float, str]] = []
if recovered_block01 is not None:
    all_blocks_rel.append(recovered_block01)

for b_lsl_s, b_lsl_e, lv in adapt_blocks:
    all_blocks_rel.append((b_lsl_s - t0, b_lsl_e - t0, lv))

all_blocks_rel.sort(key=lambda b: b[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shade_assist(ax, t: np.ndarray, assist_on: np.ndarray) -> None:
    if len(t) < 2:
        return
    dt = float(np.median(np.diff(t)))
    in_region = False
    start = 0.0
    for i in range(len(t)):
        if assist_on[i] and not in_region:
            start = t[i] - dt / 2
            in_region = True
        elif not assist_on[i] and in_region:
            ax.axvspan(start, t[i - 1] + dt / 2, color="green", alpha=0.15, zorder=1)
            in_region = False
    if in_region:
        ax.axvspan(start, t[-1] + dt / 2, color="green", alpha=0.15, zorder=1)


def _block_backgrounds(ax, blocks, t_min, t_max):
    for xs, xe, level in blocks:
        colour = _LEVEL_COLOURS.get(level, "grey")
        ax.axvspan(xs, xe, color=colour, alpha=0.07, zorder=0)
        mid_x = (xs + xe) / 2
        if t_min <= mid_x <= t_max:
            ax.text(mid_x, 0.94, level,
                    ha="center", va="top", fontsize=6,
                    color=colour, alpha=0.9,
                    transform=ax.get_xaxis_transform())


def _decorate_ax(ax, t, ph, ema, assist_on, threshold, title):
    _shade_assist(ax, t, assist_on)
    ax.plot(t, ph,  color="0.65", alpha=0.5, linewidth=0.5, label="raw P(overload)")
    ax.plot(t, ema, color="0.2",  linewidth=1.0,            label="smoothed")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="threshold")
    ax.set_ylabel("MWL  P(overload)", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(labelsize=7)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="green", alpha=0.15, label="assist ON"))
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    pct_on = 100.0 * assist_on.sum() / len(assist_on)
    ax.set_title(title + f"   |   assist ON {pct_on:.0f}%  ({assist_on.sum()}/{len(assist_on)} ticks)",
                 fontsize=9, pad=4)


def simulate_controller(ph_vals, threshold,
                        alpha=0.05, hysteresis=0.02,
                        t_hold_s=3.0, cooldown_s=0.0, step_s=0.25):
    """Recompute EMA then run state-machine. Returns (on_flags, ema_trace)."""
    ema = None; zone = None; zone_entry_t = None
    assist_on = False; cooldown_end = 0.0
    on_flags = []; ema_trace = []
    for i, v in enumerate(ph_vals):
        t = i * step_s
        ema = float(v) if ema is None else alpha * float(v) + (1.0 - alpha) * ema
        new_zone = ("above" if ema > threshold + hysteresis else
                    "below" if ema < threshold - hysteresis else "dead")
        if new_zone != zone:
            zone = new_zone; zone_entry_t = t
        hold_s = t - zone_entry_t
        if t >= cooldown_end:
            if zone == "above" and hold_s >= t_hold_s and not assist_on:
                assist_on = True
                cooldown_end = t + cooldown_s
                zone_entry_t = t
            elif zone == "below" and hold_s >= t_hold_s and assist_on:
                assist_on = False
                cooldown_end = t + cooldown_s
                zone_entry_t = t
        on_flags.append(assist_on)
        ema_trace.append(ema)
    return np.array(on_flags), np.array(ema_trace)


# Controller variants: (label, alpha, hold_s, cooldown_s)
VARIANTS = [
    ("α=0.05  hold=3s  cd=15s  (current)",  0.05, 3.0, 15.0),
    ("α=0.05  hold=3s  cd=0s   (no cooldown)", 0.05, 3.0,  0.0),
    ("α=0.10  hold=3s  cd=15s",              0.10, 3.0, 15.0),
    ("α=0.20  hold=3s  cd=15s",              0.20, 3.0, 15.0),
    ("α=0.40  hold=3s  cd=15s",              0.40, 3.0, 15.0),
]

results = []
print("\nController variants:")
print(f"  {'Label':<38}  {'assist_ON%':>10}  {'ON→OFF cycles':>13}")
for label, alpha, hold_s, cd_s in VARIANTS:
    on_flags, ema_trace = simulate_controller(
        ph_new, thr_full, alpha=alpha, t_hold_s=hold_s, cooldown_s=cd_s)
    pct = 100.0 * on_flags.sum() / len(on_flags)
    cycles = int((np.diff(on_flags.astype(int)) == 1).sum())
    print(f"  {label:<38}  {pct:9.1f}%  {cycles:13d}")
    results.append((label, on_flags, ema_trace))

# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------

OUT_PATH_VARIANTS = OUT_PATH.parent / "adaptation_pself__fig01__mwl_timeline_ema_variants.png"

n = len(VARIANTS)
fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=True)
fig.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.98, hspace=0.45)

for ax, (label, on_flags, ema_trace) in zip(axes, results):
    _block_backgrounds(ax, all_blocks_rel, t_rel[0], t_rel[-1])
    _decorate_ax(ax, t_rel, ph_new, ema_trace, on_flags, thr_full, label)

axes[-1].set_xlabel("Scenario time (s)", fontsize=9)

OUT_PATH_VARIANTS.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH_VARIANTS, dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_PATH_VARIANTS}")

# Also update the single-panel figure (current params)
dur_min = (t_rel[-1] - t_rel[0]) / 60
fig2, ax2 = plt.subplots(figsize=(14, 5))
fig2.subplots_adjust(top=0.87, bottom=0.10, left=0.06, right=0.98)
_block_backgrounds(ax2, all_blocks_rel, t_rel[0], t_rel[-1])
_decorate_ax(ax2, t_rel, ph_new, ema_new, on_new, thr_full,
             f"MWL Adaptation Timeline  (+block01 model, thr={thr_full:.3f})  —  {dur_min:.1f} min")
ax2.set_xlabel("Scenario time (s)", fontsize=9)
fig2.savefig(OUT_PATH, dpi=150)
plt.close(fig2)
print(f"Saved: {OUT_PATH}")
