"""Synthesise an audit CSV from the +block_01 model cache and run the
post-session analysis + plot on it.

No model retraining — reads pre-computed arrays written by
_tmp_retrain_with_block01.py.

Run:
    .\.venv\Scripts\Activate.ps1
    python scripts/_tmp_analyse_new_model.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE_DIR   = REPO / "results" / "_tmp_new_model_cache"
ORIG_AUDIT  = Path(r"C:\data\adaptive_matb\openmatb\PSELF\S005\mwl_audit.csv")
SESSION_CSV = Path(r"C:\data\adaptive_matb\openmatb\PSELF\S005\sessions\2026-04-08\7_260408_142612.csv")
SYNTH_AUDIT = REPO / "results" / "_tmp_new_model_cache" / "synth_audit_new_model.csv"
OUT_FIG     = REPO / "results" / "figures" / "adaptation_pself__fig01__mwl_timeline_new_model.png"
OUT_JSON    = REPO / "results" / "_tmp_new_model_cache" / "analysis_new_model.json"

# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------

arrays  = np.load(CACHE_DIR / "adapt_inference.npz")
ph_new  = arrays["ph_new"]
ema_new = arrays["ema_new"]
lsl_ts  = arrays["lsl_ts"]

meta        = json.load(open(CACHE_DIR / "meta.json"))
thr_full    = float(meta["threshold"])

print(f"Cache loaded: {len(ph_new)} windows, threshold={thr_full:.4f}")

# ---------------------------------------------------------------------------
# Derive scenario_time_s from original audit's LSL→scenario offset
# ---------------------------------------------------------------------------

with open(ORIG_AUDIT, newline="") as f:
    first_row = next(csv.DictReader(f))

lsl_origin     = float(first_row["timestamp_lsl"])
scenario_origin = float(first_row["scenario_time_s"])
lsl_t0_scenario = lsl_origin - scenario_origin   # LSL ts when scenario t=0

scenario_times = lsl_ts - lsl_t0_scenario       # scenario_time_s for each window

print(f"Scenario time range: {scenario_times[0]:.1f}s → {scenario_times[-1]:.1f}s")

# ---------------------------------------------------------------------------
# Re-run state machine, tracking all audit fields
# ---------------------------------------------------------------------------

ALPHA      = 0.05
HYSTERESIS = 0.02
T_HOLD_S   = 3.0
COOLDOWN_S = 15.0
STEP_S     = 0.25

ema_state    = None
zone         = None
zone_entry_t = None
assist_on    = False
cooldown_end = 0.0
hold_counter = 0.0

rows: list[dict] = []

for i, (lsl_t, sc_t, raw, smoothed) in enumerate(
    zip(lsl_ts, scenario_times, ph_new, ema_new)
):
    # EMA already computed in cache — use it directly
    ema_val = float(smoothed)

    new_zone = (
        "above" if ema_val > thr_full + HYSTERESIS else
        "below" if ema_val < thr_full - HYSTERESIS else
        "dead"
    )
    if new_zone != zone:
        zone = new_zone
        zone_entry_t = sc_t
        hold_counter = 0.0
    else:
        hold_counter = sc_t - zone_entry_t

    action = "hold"
    reason = "hold"
    cooldown_remaining = max(0.0, cooldown_end - sc_t)

    if sc_t >= cooldown_end:
        if zone == "above" and hold_counter >= T_HOLD_S and not assist_on:
            assist_on = True
            cooldown_end = sc_t + COOLDOWN_S
            zone_entry_t = sc_t
            hold_counter = 0.0
            cooldown_remaining = COOLDOWN_S
            action = "assist_on"
            reason = f"EMA {ema_val:.3f} > thr+hyst {thr_full+HYSTERESIS:.3f} for {T_HOLD_S}s"
        elif zone == "below" and hold_counter >= T_HOLD_S and assist_on:
            assist_on = False
            cooldown_end = sc_t + COOLDOWN_S
            zone_entry_t = sc_t
            hold_counter = 0.0
            cooldown_remaining = COOLDOWN_S
            action = "assist_off"
            reason = f"EMA {ema_val:.3f} < thr-hyst {thr_full-HYSTERESIS:.3f} for {T_HOLD_S}s"
        else:
            reason = (
                f"cooldown_ok zone={zone} hold={hold_counter:.1f}/{T_HOLD_S}s"
                if zone != "dead" else "dead_zone"
            )
    else:
        reason = f"cooldown {cooldown_remaining:.1f}s remaining"

    rows.append({
        "timestamp_lsl":        f"{lsl_t:.6f}",
        "scenario_time_s":      f"{sc_t:.3f}",
        "mwl_raw":              f"{raw:.6f}",
        "mwl_smoothed":         f"{ema_val:.6f}",
        "signal_quality":       "1.000",
        "threshold":            f"{thr_full:.4f}",
        "action":               action,
        "assistance_on":        str(assist_on),
        "cooldown_remaining_s": f"{cooldown_remaining:.2f}",
        "hold_counter_s":       f"{hold_counter:.2f}",
        "reason":               reason,
    })

COLUMNS = [
    "timestamp_lsl", "scenario_time_s", "mwl_raw", "mwl_smoothed",
    "signal_quality", "threshold", "action", "assistance_on",
    "cooldown_remaining_s", "hold_counter_s", "reason",
]

SYNTH_AUDIT.parent.mkdir(parents=True, exist_ok=True)
with open(SYNTH_AUDIT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

n_on  = sum(1 for r in rows if r["action"] == "assist_on")
n_off = sum(1 for r in rows if r["action"] == "assist_off")
pct   = 100 * sum(1 for r in rows if r["assistance_on"] == "True") / len(rows)
print(f"Synthetic audit written: {SYNTH_AUDIT.name}")
print(f"  {len(rows)} rows | assist_on events={n_on} | assist_off events={n_off} | {pct:.1f}% time ON")

# ---------------------------------------------------------------------------
# Run analyse_adaptation_session.py
# ---------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Running analyse_adaptation_session.py ...")
result = subprocess.run(
    [
        sys.executable,
        str(REPO / "scripts" / "analyse_adaptation_session.py"),
        "--audit",   str(SYNTH_AUDIT),
        "--session", str(SESSION_CSV),
        "--out",     str(OUT_JSON),
    ],
    capture_output=False,
)
if result.returncode != 0:
    print("ERROR: analyse_adaptation_session.py failed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Run plot_adaptation_session.py
# ---------------------------------------------------------------------------

print("\n" + "-" * 60)
print("Running plot_adaptation_session.py ...")
result = subprocess.run(
    [
        sys.executable,
        str(REPO / "scripts" / "plot_adaptation_session.py"),
        "--audit",   str(SYNTH_AUDIT),
        "--session", str(SESSION_CSV),
        "--out",     str(OUT_FIG),
    ],
    capture_output=False,
)
if result.returncode != 0:
    print("ERROR: plot_adaptation_session.py failed.")
    sys.exit(1)

print(f"\nFigure: {OUT_FIG}")
print(f"JSON:   {OUT_JSON}")
