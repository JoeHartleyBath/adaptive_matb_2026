"""verify_adaptation_session.py

Checks a completed staircase-calibration session CSV and reports whether
the adaptation system ran correctly.

Usage (quick — auto-finds latest session for a participant):
    python scripts/verify_adaptation_session.py --participant PDEV

Usage (explicit CSV):
    python scripts/verify_adaptation_session.py --csv "C:/data/.../1_xxx.csv"

Exit codes:
    0  all required checks passed
    1  one or more checks FAILED
    2  no CSV found / argument error
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\u2713"
FAIL = "\u2717"
WARN = "!"


def _find_latest_csv(output_root: str, participant: str) -> Optional[Path]:
    base = Path(output_root) / "openmatb" / participant
    if not base.exists():
        return None
    csvs = sorted(base.rglob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Skip any manifest-style files; find first proper CSV with header
    for p in csvs:
        if p.stat().st_size > 0:
            return p
    return None


def _load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _adaptation_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("type") == "adaptation"]


def _parse_adaptation_json(row: dict) -> Optional[dict]:
    try:
        return json.loads(row.get("value", ""))
    except (json.JSONDecodeError, TypeError):
        return None


def _performance_rows(rows: list[dict], module: str, metric: str) -> list[dict]:
    return [
        r for r in rows
        if r.get("type") == "performance"
        and r.get("module") == module
        and r.get("address") == metric
    ]


def _scenario_duration(rows: list[dict]) -> Optional[float]:
    """Last scenario_time value in the log."""
    for row in reversed(rows):
        try:
            return float(row["scenario_time"])
        except (KeyError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_checks(rows: list[dict], csv_path: Path) -> int:
    """Run all checks; return number of failures."""
    failures = 0
    warnings = 0

    print(f"\n{'='*65}")
    print(f"  ADAPTATION SESSION VERIFICATION")
    print(f"  CSV: {csv_path.name}")
    print(f"{'='*65}\n")

    # --- basic session info ------------------------------------------------
    duration = _scenario_duration(rows)
    print(f"Session duration : {duration:.1f} s" if duration else "Session duration : unknown")
    print(f"Total log rows   : {len(rows)}")

    # -------------------------------------------------------------------
    # CHECK 1: adaptation_init row present
    # -------------------------------------------------------------------
    adapt_rows = _adaptation_rows(rows)
    init_rows = [r for r in adapt_rows if _parse_adaptation_json(r) and
                 _parse_adaptation_json(r).get("event") == "adaptation_init"]

    ok = bool(init_rows)
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 1: adaptation_init logged")
    if not ok:
        failures += 1
        print("       No 'adaptation_init' row found in CSV.")
        print("       Likely causes:")
        print("         - Session was run WITHOUT --adaptation flag")
        print("         - AdaptationScheduler crashed before setup completed")
        print("         - Bootstrap injection failed (check run args)")
        # Without init, nothing else can be verified — bail early.
        print(f"\n{'='*65}")
        print(f"  RESULT: {failures} check(s) FAILED  (early exit — init missing)")
        print(f"{'='*65}\n")
        return failures

    init_data = _parse_adaptation_json(init_rows[0])
    cfg = init_data.get("config", {})
    params = init_data.get("initial_params", {})
    print(f"       d_init={cfg.get('d_init')}  seed={cfg.get('seed')}")
    print(f"       target_score={cfg.get('target_score')}  window_sec={cfg.get('window_sec')}")
    print(f"       initial d={params.get('d'):.3f}" if isinstance(params.get('d'), float) else "")

    # -------------------------------------------------------------------
    # CHECK 2: performance data collected (track cursor_in_target)
    # -------------------------------------------------------------------
    track_perf = _performance_rows(rows, "track", "cursor_in_target")
    ok = len(track_perf) >= 5
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 2: tracking performance samples collected")
    print(f"       cursor_in_target rows: {len(track_perf)}")
    if not ok:
        failures += 1
        print("       Too few samples — staircase window will never trigger.")
    else:
        values = [float(r["value"]) for r in track_perf if r.get("value") != ""]
        if values:
            mean_score = sum(values) / len(values)
            print(f"       mean cursor_in_target: {mean_score:.3f}")

    # -------------------------------------------------------------------
    # CHECK 3: at least one staircase step fired
    # -------------------------------------------------------------------
    step_rows = [r for r in adapt_rows if _parse_adaptation_json(r) and
                 _parse_adaptation_json(r).get("event") == "adaptation_step"]

    if not step_rows:
        # Not necessarily a failure if session was very short
        window_sec = cfg.get("window_sec", 45)
        if duration and duration < window_sec:
            marker = WARN
            warnings += 1
            print(f"\n[{marker}] CHECK 3: staircase steps fired  (WARNING — session shorter than window)")
            print(f"       Session {duration:.0f}s < window {window_sec}s — no step expected yet.")
        else:
            marker = FAIL
            failures += 1
            print(f"\n[{marker}] CHECK 3: staircase steps fired")
            print(f"       No adaptation_step rows found despite session being long enough.")
            print(f"       Check min_samples ({cfg.get('min_samples')}) and cooldown settings.")
    else:
        print(f"\n[{PASS}] CHECK 3: staircase steps fired")
        print(f"       Total steps: {len(step_rows)}")

        d_values = []
        for r in step_rows:
            d = _parse_adaptation_json(r)
            if d and "d_new" in d:
                d_values.append(d["d_new"])
            elif d and "d" in d:
                d_values.append(d["d"])

        if d_values:
            print(f"       d trajectory: {cfg.get('d_init')} → "
                  + " → ".join(f"{v:.3f}" for v in d_values))

        # up vs down steps
        delta_vals = []
        for r in step_rows:
            d = _parse_adaptation_json(r)
            if d:
                delta = d.get("delta")
                if delta is not None:
                    delta_vals.append(float(delta))
        if delta_vals:
            ups = sum(1 for v in delta_vals if v > 0)
            downs = sum(1 for v in delta_vals if v < 0)
            print(f"       Steps up: {ups}  |  Steps down: {downs}")

    # -------------------------------------------------------------------
    # CHECK 4: d stayed within [d_min, d_max]
    # -------------------------------------------------------------------
    if step_rows and d_values:
        d_min = cfg.get("d_min", 0.0)
        d_max = cfg.get("d_max", 1.0)
        out_of_bounds = [v for v in d_values if v < d_min or v > d_max]
        ok = not out_of_bounds
        marker = PASS if ok else FAIL
        print(f"\n[{marker}] CHECK 4: d stayed within [{d_min}, {d_max}]")
        if not ok:
            failures += 1
            print(f"       Out-of-bounds values: {out_of_bounds}")
    else:
        print(f"\n[-] CHECK 4: d bounds  (skipped — no step data)")

    # -------------------------------------------------------------------
    # CHECK 5: no adaptation_error rows
    # -------------------------------------------------------------------
    error_rows = [r for r in adapt_rows if _parse_adaptation_json(r) and
                  _parse_adaptation_json(r).get("event") == "adaptation_error"]
    ok = not error_rows
    marker = PASS if ok else FAIL
    print(f"\n[{marker}] CHECK 5: no adaptation_error rows")
    if not ok:
        failures += 1
        for r in error_rows:
            print(f"       {r.get('value', '')[:120]}")

    # -------------------------------------------------------------------
    # CHECK 6: session ended cleanly (manual end row present)
    # -------------------------------------------------------------------
    end_rows = [r for r in rows if r.get("type") == "manual" and
                str(r.get("value", "")).strip() == "end"]
    ok = bool(end_rows)
    marker = PASS if ok else WARN
    print(f"\n[{marker}] CHECK 6: session ended cleanly (manual 'end' row)")
    if not ok:
        warnings += 1
        print("       Missing manual end row — session may have crashed.")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    total_adapt_rows = len(adapt_rows)
    print(f"  Adaptation log rows   : {total_adapt_rows}")
    print(f"  Steps fired           : {len(step_rows)}")
    print(f"  Tracking perf samples : {len(track_perf)}")
    if warnings:
        print(f"  Warnings              : {warnings}")
    status = "PASSED" if failures == 0 else f"FAILED ({failures} check(s))"
    print(f"\n  RESULT: {status}")
    print(f"{'='*65}\n")

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a staircase-calibration session CSV."
    )
    parser.add_argument("--csv", help="Direct path to the session CSV file.")
    parser.add_argument("--participant", help="Participant ID (e.g. PDEV). Auto-finds latest session.")
    parser.add_argument(
        "--output-root",
        default=r"C:\data\adaptive_matb",
        help="Root output directory (default: C:\\data\\adaptive_matb)",
    )
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    elif args.participant:
        csv_path = _find_latest_csv(args.output_root, args.participant)
        if csv_path is None:
            print(f"ERROR: No CSV found for participant '{args.participant}' under {args.output_root}",
                  file=sys.stderr)
            return 2
        print(f"Auto-selected: {csv_path}")
    else:
        parser.print_help()
        return 2

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    rows = _load_csv(csv_path)
    failures = run_checks(rows, csv_path)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
