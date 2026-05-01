"""Temporary: per-block performance review for PSELF run 8 (full_calibration_pself_c1)."""
import csv
from pathlib import Path

CSV = Path(r"C:\data\adaptive_matb\openmatb\PSELF\STEST\sessions\2026-04-09\8_260409_153232.csv")

# Block order (condition 1): H L M L M H M H L
BLOCK_ORDER = ["HIGH", "LOW", "MODERATE", "LOW", "MODERATE", "HIGH", "MODERATE", "HIGH", "LOW"]
LSL_SETTLE = 5
BLOCK_DUR  = 60

blocks = []
for i, level in enumerate(BLOCK_ORDER):
    t0 = LSL_SETTLE + i * BLOCK_DUR
    t1 = t0 + BLOCK_DUR
    blocks.append({
        "level": level, "t0": t0, "t1": t1,
        "track_in": 0, "track_total": 0,
        "sysmon_hit": 0, "sysmon_miss": 0,
        "resman_a": [], "resman_b": [],
        "comms_needed": 0, "comms_hit": 0,
    })

def block_for(t):
    for b in blocks:
        if b["t0"] <= t < b["t1"]:
            return b
    return None

# Buffer for comms grouping (rows arrive in triplets at same timestamp)
_comms_buf: dict[str, dict] = {}

with open(CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            t = float(row["scenario_time"])
        except ValueError:
            continue
        module  = row["module"]
        address = row["address"]
        value   = row["value"]
        b = block_for(t)
        if b is None:
            continue

        if module == "track" and address == "cursor_in_target":
            try:
                b["track_in"]    += float(value)
                b["track_total"] += 1
            except ValueError:
                pass

        elif module == "sysmon" and address == "signal_detection":
            if value == "HIT":
                b["sysmon_hit"]  += 1
            elif value == "MISS":
                b["sysmon_miss"] += 1

        elif module == "resman":
            if address == "a_in_tolerance":
                try: b["resman_a"].append(float(value))
                except ValueError: pass
            elif address == "b_in_tolerance":
                try: b["resman_b"].append(float(value))
                except ValueError: pass

        elif module == "communications":
            key = f"{t:.3f}"
            if key not in _comms_buf:
                _comms_buf[key] = {"needed": None, "target": None, "responded": None, "b": b}
            entry = _comms_buf[key]
            if address == "response_was_needed":
                try: entry["needed"] = int(float(value))
                except ValueError: pass
            elif address == "target_radio":
                entry["target"] = value
            elif address == "responded_radio":
                entry["responded"] = value
            # Flush if complete
            if all(v is not None for v in (entry["needed"], entry["target"], entry["responded"])):
                if entry["needed"] == 1:
                    entry["b"]["comms_needed"] += 1
                    if entry["target"] == entry["responded"]:
                        entry["b"]["comms_hit"] += 1
                del _comms_buf[key]

def nanmean(vals):
    valid = [v for v in vals if v == v]  # exclude NaN
    return sum(valid) / len(valid) if valid else float("nan")

# Print table
print(f"\n{'Blk':<4} {'Level':<10} {'Track':>6} {'Sysmon':>7} {'Comms':>6} {'ResA':>5} {'ResB':>5} {'Composite':>10}")
print("-" * 60)
level_agg: dict[str, list] = {"LOW": [], "MODERATE": [], "HIGH": []}

for i, b in enumerate(blocks, 1):
    trk  = (b["track_in"] / b["track_total"])      if b["track_total"] else float("nan")
    det  = b["sysmon_hit"] + b["sysmon_miss"]
    sys_ = (b["sysmon_hit"] / det)                 if det              else float("nan")
    com  = (b["comms_hit"] / b["comms_needed"])    if b["comms_needed"] else float("nan")
    ra   = (sum(b["resman_a"]) / len(b["resman_a"])) if b["resman_a"] else float("nan")
    rb   = (sum(b["resman_b"]) / len(b["resman_b"])) if b["resman_b"] else float("nan")
    # Resman composite: mean of tanks A and B
    res  = nanmean([ra, rb])
    # Overall composite: equal-weight mean of available subtask scores
    # Comms excluded from a block's mean when no prompt was presented (NaN)
    composite = nanmean([trk, sys_, com, res])

    def pct(v): return f"{v*100:5.1f}" if v == v else "  nan"
    print(f"{i:<4} {b['level']:<10} {pct(trk)} {pct(sys_)} {pct(com)} {pct(ra)} {pct(rb)}  {pct(composite)}")
    level_agg[b["level"]].append((trk, sys_, com, res, composite))

print("-" * 60)
print("\nMeans by level (3 blocks each):")
print(f"{'Level':<10} {'Track':>6} {'Sysmon':>7} {'Comms':>6} {'Resman':>7} {'Composite':>10}")
print("-" * 52)
for lvl in ("LOW", "MODERATE", "HIGH"):
    rows = level_agg[lvl]
    if not rows:
        continue
    cols = list(zip(*rows))
    vals = [nanmean(c) * 100 for c in cols]
    print(f"{lvl:<10} {vals[0]:>6.1f} {vals[1]:>7.1f} {vals[2]:>6.1f} {vals[3]:>7.1f} {vals[4]:>10.1f}")

