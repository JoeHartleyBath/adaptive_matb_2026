"""verify_mwl_adaptation_session.py

Post-hoc verification of an MWL-driven adaptation session.

Reads the audit CSV produced by AdaptationLogger and checks that
the MWL controller behaved correctly: binary tracking assistance
toggles, cooldown enforcement, and signal quality gating.

╔══════════════════════════════════════════════════════════════════════╗
║  The adaptation is a BINARY tracking-only toggle (assist_on /      ║
║  assist_off).  There is NO graduated difficulty scalar (d).        ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python tests/verification/verify_mwl_adaptation_session.py --csv path/to/mwl_audit.csv

Exit codes:
    0  all checks passed
    1  one or more checks FAILED
    2  argument error / file not found
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PASS = "\u2713"
FAIL = "\u2717"
WARN = "!"

EXPECTED_COLUMNS = [
    "timestamp_lsl",
    "scenario_time_s",
    "mwl_raw",
    "mwl_smoothed",
    "signal_quality",
    "threshold",
    "action",
    "assistance_on",
    "cooldown_remaining_s",
    "hold_counter_s",
    "reason",
]


def _load_audit_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def run_checks(rows: list[dict], csv_path: Path) -> int:
    failures = 0

    print(f"\n{'='*65}")
    print(f"  MWL ADAPTATION SESSION VERIFICATION")
    print(f"  (binary tracking-only assistance toggle)")
    print(f"  CSV: {csv_path.name}")
    print(f"{'='*65}\n")

    # -------------------------------------------------------------------
    # CHECK 1: audit CSV is non-empty and has expected schema
    # -------------------------------------------------------------------
    if not rows:
        print(f"[{FAIL}] CHECK 1: audit CSV contains data")
        print("       File is empty — no decision ticks logged.")
        print(f"\n{'='*65}")
        print(f"  RESULT: FAILED (1 check)")
        print(f"{'='*65}\n")
        return 1

    actual_cols = list(rows[0].keys())
    ok = actual_cols == EXPECTED_COLUMNS
    marker = PASS if ok else FAIL
    print(f"[{marker}] CHECK 1: schema matches expected columns")
    print(f"       Rows: {len(rows)}")
    if not ok:
        failures += 1
        print(f"       Expected: {EXPECTED_COLUMNS}")
        print(f"       Got:      {actual_cols}")

    # Parse numeric fields once
    for r in rows:
        for key in ("scenario_time_s", "mwl_raw", "mwl_smoothed", "signal_quality",
                     "threshold", "cooldown_remaining_s",
                     "hold_counter_s", "timestamp_lsl"):
            try:
                r[key] = float(r[key])
            except (ValueError, KeyError):
                pass
        # Parse assistance_on as boolean
        if isinstance(r.get("assistance_on"), str):
            r["assistance_on"] = r["assistance_on"].strip().lower() in ("true", "1")

    duration = rows[-1]["scenario_time_s"] - rows[0]["scenario_time_s"]
    print(f"       Duration: {duration:.1f}s  ({rows[0]['scenario_time_s']:.1f}s → {rows[-1]['scenario_time_s']:.1f}s)")

    # -------------------------------------------------------------------
    # CHECK 2: action values are valid
    # -------------------------------------------------------------------
    valid_actions = {"hold", "assist_on", "assist_off"}
    actions = {r["action"] for r in rows}
    unknown = actions - valid_actions
    ok = not unknown
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 2: all action values are valid")
    if not ok:
        failures += 1
        print(f"       Unknown actions: {unknown}")
    else:
        counts = {a: sum(1 for r in rows if r["action"] == a) for a in valid_actions}
        print(f"       hold={counts.get('hold', 0)}  "
              f"assist_on={counts.get('assist_on', 0)}  "
              f"assist_off={counts.get('assist_off', 0)}")

    # -------------------------------------------------------------------
    # CHECK 3: assistance_on is consistent with actions
    #   After assist_on  → assistance_on must be True
    #   After assist_off → assistance_on must be False
    # -------------------------------------------------------------------
    toggle_rows = [r for r in rows if r["action"] in ("assist_on", "assist_off")]
    inconsistent = []
    for r in toggle_rows:
        expected = r["action"] == "assist_on"
        if r["assistance_on"] != expected:
            inconsistent.append(r)
    ok = not inconsistent
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 3: assistance_on consistent with actions")
    if not ok:
        failures += 1
        print(f"       Inconsistencies: {len(inconsistent)}")
        for r in inconsistent[:3]:
            print(f"         t={r['scenario_time_s']:.1f}  action={r['action']}  "
                  f"assistance_on={r['assistance_on']}")

    # -------------------------------------------------------------------
    # CHECK 4: assist_on only fires when MWL is above threshold,
    #          assist_off only fires when MWL is below threshold
    # -------------------------------------------------------------------
    wrong_direction = []
    for r in rows:
        if not isinstance(r["mwl_smoothed"], float) or not isinstance(r["threshold"], float):
            continue
        if r["action"] == "assist_on" and r["mwl_smoothed"] < r["threshold"]:
            wrong_direction.append(r)
        elif r["action"] == "assist_off" and r["mwl_smoothed"] > r["threshold"]:
            wrong_direction.append(r)
    ok = not wrong_direction
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 4: toggle direction matches MWL vs threshold")
    if not ok:
        failures += 1
        print(f"       Wrong-direction actions: {len(wrong_direction)}")
        for r in wrong_direction[:3]:
            print(f"         t={r['scenario_time_s']:.1f}  action={r['action']}  "
                  f"smoothed={r['mwl_smoothed']:.3f}  threshold={r['threshold']:.3f}")

    # -------------------------------------------------------------------
    # CHECK 5: no double-toggle (assist_on when already on,
    #          assist_off when already off)
    # -------------------------------------------------------------------
    double_toggles = []
    state = False  # initial assistance state
    for r in rows:
        if r["action"] == "assist_on":
            if state:
                double_toggles.append(r)
            state = True
        elif r["action"] == "assist_off":
            if not state:
                double_toggles.append(r)
            state = False
    ok = not double_toggles
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 5: no redundant toggles (double on/off)")
    if not ok:
        failures += 1
        print(f"       Double-toggles: {len(double_toggles)}")
        for r in double_toggles[:3]:
            print(f"         t={r['scenario_time_s']:.1f}  action={r['action']}")

    # -------------------------------------------------------------------
    # CHECK 6: cooldown is respected between toggles
    # -------------------------------------------------------------------
    action_rows = [r for r in rows if r["action"] in ("assist_on", "assist_off")]
    cooldown_violations = []
    for i in range(1, len(action_rows)):
        gap = action_rows[i]["scenario_time_s"] - action_rows[i - 1]["scenario_time_s"]
        # Cooldown default is 15s; allow small tolerance for tick quantisation
        if gap < 14.5:
            cooldown_violations.append((action_rows[i - 1], action_rows[i], gap))

    if action_rows:
        ok = not cooldown_violations
        marker = PASS if ok else FAIL
        print(f"\n[{marker}] CHECK 6: cooldown respected between toggles")
        if not ok:
            failures += 1
            for prev_r, cur_r, gap in cooldown_violations[:3]:
                print(f"         t={prev_r['scenario_time_s']:.1f} → {cur_r['scenario_time_s']:.1f}  "
                      f"gap={gap:.1f}s (< 15s)")
        else:
            gaps = [action_rows[i]["scenario_time_s"] - action_rows[i - 1]["scenario_time_s"]
                    for i in range(1, len(action_rows))]
            if gaps:
                print(f"       Min gap between toggles: {min(gaps):.1f}s")
    else:
        print(f"\n[-] CHECK 6: cooldown  (skipped — no toggles fired)")

    # -------------------------------------------------------------------
    # CHECK 7: no toggle when signal quality is below minimum
    # -------------------------------------------------------------------
    low_quality_actions = [
        r for r in rows
        if r["action"] in ("assist_on", "assist_off")
        and isinstance(r["signal_quality"], float)
        and r["signal_quality"] < 0.5
    ]
    ok = not low_quality_actions
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 7: no toggle during low signal quality")
    if not ok:
        failures += 1
        print(f"       Toggles with quality < 0.5: {len(low_quality_actions)}")

    # -------------------------------------------------------------------
    # CHECK 8: timestamps are monotonically increasing
    # -------------------------------------------------------------------
    non_monotonic = []
    for i in range(1, len(rows)):
        if isinstance(rows[i]["scenario_time_s"], float) and isinstance(rows[i-1]["scenario_time_s"], float):
            if rows[i]["scenario_time_s"] < rows[i-1]["scenario_time_s"] - 1e-6:
                non_monotonic.append(i)
    ok = not non_monotonic
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 8: timestamps are monotonically increasing")
    if not ok:
        failures += 1
        print(f"       Non-monotonic at rows: {non_monotonic[:5]}")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    n_toggles = len(action_rows)
    toggle_seq = []
    for r in action_rows:
        label = "ON" if r["action"] == "assist_on" else "OFF"
        toggle_seq.append(f"{label}@{r['scenario_time_s']:.0f}s")

    print(f"\n{'='*65}")
    print(f"  Total decision ticks  : {len(rows)}")
    print(f"  Toggles fired         : {n_toggles}")
    if toggle_seq:
        print(f"  Toggle sequence       : " + " → ".join(toggle_seq))
    status = "PASSED" if failures == 0 else f"FAILED ({failures} check(s))"
    print(f"\n  RESULT: {status}")
    print(f"{'='*65}\n")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an MWL adaptation session audit CSV.",
    )
    parser.add_argument("--csv", required=True, help="Path to the mwl_audit.csv file.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    rows = _load_audit_csv(csv_path)
    failures = run_checks(rows, csv_path)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
