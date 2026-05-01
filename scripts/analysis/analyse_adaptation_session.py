"""Post-session analysis of an MWL-driven adaptation run.

Loads the adaptation audit CSV (from adaptation_logger.py) and the
OpenMATB session CSV, aligns them by scenario time, and computes:
  - Switching dynamics (detection latency, toggles by block type)
  - MWL trajectory statistics (mean/std by assistance state)
  - Task performance by assistance state
  - Summary report (JSON + console)

Usage
-----
    python scripts/analysis/analyse_adaptation_session.py \\
        --audit   /path/to/adaptation_audit.csv \\
        --session /path/to/openmatb_session.csv \\
        --out     results/P001/adaptation_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from adaptation.audit_loader import (  # noqa: E402
    AuditRow,
    BlockSegment,
    load_audit_csv,
    load_session_blocks,
)



def assign_rows_to_blocks(
    rows: list[AuditRow],
    blocks: list[BlockSegment],
) -> dict[str, list[AuditRow]]:
    """Map each audit row to the block it falls within (by scenario_time_s).

    Returns dict keyed by block name.  Rows outside any block are stored
    under the key "__unassigned__".
    """
    by_block: dict[str, list[AuditRow]] = {"__unassigned__": []}
    for b in blocks:
        by_block[b.name] = []

    for row in rows:
        assigned = False
        for b in blocks:
            if b.start_sec <= row.scenario_time_s <= b.end_sec:
                by_block[b.name].append(row)
                assigned = True
                break
        if not assigned:
            by_block["__unassigned__"].append(row)

    return by_block


# ---------------------------------------------------------------------------
# Switching dynamics (subtask 2)
# ---------------------------------------------------------------------------

def _toggle_episodes(rows: list[AuditRow]) -> list[dict]:
    """Extract assist-ON episodes from a sequence of audit rows.

    An episode starts at an ``assist_on`` action and ends at either:
      - the next ``assist_off`` action, or
      - the end of the row sequence (open episode).

    Returns list of dicts with keys: start_s, end_s, duration_s, open.
    """
    episodes: list[dict] = []
    ep_start: float | None = None

    for row in rows:
        if row.action == "assist_on":
            ep_start = row.scenario_time_s
        elif row.action == "assist_off" and ep_start is not None:
            episodes.append({
                "start_s": ep_start,
                "end_s": row.scenario_time_s,
                "duration_s": row.scenario_time_s - ep_start,
                "open": False,
            })
            ep_start = None

    # If still open at end of sequence
    if ep_start is not None and rows:
        episodes.append({
            "start_s": ep_start,
            "end_s": rows[-1].scenario_time_s,
            "duration_s": rows[-1].scenario_time_s - ep_start,
            "open": True,
        })

    return episodes


def compute_switching_dynamics(
    blocks: list[BlockSegment],
    rows_by_block: dict[str, list[AuditRow]],
) -> dict:
    """Compute switching dynamics per block and per level category.

    Returns a dict with:
      - per_block: list of per-block summaries
      - by_level: aggregated stats for HIGH / MODERATE / LOW
      - overall: session-wide toggle count and episode durations
    """
    per_block: list[dict] = []
    by_level: dict[str, list[dict]] = {"HIGH": [], "MODERATE": [], "LOW": []}

    for block in blocks:
        block_rows = rows_by_block.get(block.name, [])
        toggles_on = [r for r in block_rows if r.action == "assist_on"]
        toggles_off = [r for r in block_rows if r.action == "assist_off"]
        episodes = _toggle_episodes(block_rows)

        # Detection latency: time from block start to first assist_on
        detection_latency: float | None = None
        if toggles_on:
            detection_latency = toggles_on[0].scenario_time_s - block.start_sec

        episode_durations = [e["duration_s"] for e in episodes if not e["open"]]

        summary = {
            "block_name": block.name,
            "level": block.level,
            "block_num": block.block_num,
            "start_sec": block.start_sec,
            "end_sec": block.end_sec,
            "duration_sec": block.end_sec - block.start_sec,
            "n_assist_on": len(toggles_on),
            "n_assist_off": len(toggles_off),
            "n_episodes": len(episodes),
            "detection_latency_s": detection_latency,
            "episode_durations_s": episode_durations,
            "n_ticks": len(block_rows),
        }
        per_block.append(summary)

        if block.level in by_level:
            by_level[block.level].append(summary)

    # Aggregate by level
    level_agg: dict[str, dict] = {}
    for level, summaries in by_level.items():
        if not summaries:
            level_agg[level] = {"n_blocks": 0}
            continue

        latencies = [s["detection_latency_s"] for s in summaries
                     if s["detection_latency_s"] is not None]
        all_durations = []
        for s in summaries:
            all_durations.extend(s["episode_durations_s"])
        total_triggers = sum(s["n_assist_on"] for s in summaries)

        agg: dict = {
            "n_blocks": len(summaries),
            "total_triggers": total_triggers,
            "triggers_per_block": [s["n_assist_on"] for s in summaries],
        }
        if latencies:
            agg["detection_latency_mean_s"] = float(np.mean(latencies))
            agg["detection_latency_std_s"] = float(np.std(latencies))
            agg["detection_latency_all_s"] = latencies
        if all_durations:
            agg["episode_duration_mean_s"] = float(np.mean(all_durations))
            agg["episode_duration_std_s"] = float(np.std(all_durations))

        level_agg[level] = agg

    # Overall
    all_rows = []
    for block in blocks:
        all_rows.extend(rows_by_block.get(block.name, []))
    all_rows.sort(key=lambda r: r.scenario_time_s)
    all_episodes = _toggle_episodes(all_rows)
    closed_durations = [e["duration_s"] for e in all_episodes if not e["open"]]

    overall = {
        "total_toggles_on": sum(s["n_assist_on"] for s in per_block),
        "total_toggles_off": sum(s["n_assist_off"] for s in per_block),
        "total_episodes": len(all_episodes),
        "correct_detections_HIGH": level_agg.get("HIGH", {}).get("total_triggers", 0),
        "ambiguous_triggers_MODERATE": level_agg.get("MODERATE", {}).get("total_triggers", 0),
        "false_alarms_LOW": level_agg.get("LOW", {}).get("total_triggers", 0),
    }
    if closed_durations:
        overall["episode_duration_mean_s"] = float(np.mean(closed_durations))
        overall["episode_duration_std_s"] = float(np.std(closed_durations))

    return {
        "per_block": per_block,
        "by_level": level_agg,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# MWL trajectory statistics (subtask 3)
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict:
    """Return mean, std, min, max, median for a list of floats."""
    if not values:
        return {}
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n": len(values),
    }


def compute_mwl_trajectory(
    audit_rows: list[AuditRow],
    blocks: list[BlockSegment],
    rows_by_block: dict[str, list[AuditRow]],
) -> dict:
    """Compute MWL trajectory statistics.

    Returns:
      - by_assistance_state: MWL stats when assist ON vs OFF
      - by_level: MWL stats per workload level (HIGH/MODERATE/LOW)
      - time_above_threshold: fraction of ticks where smoothed MWL >= threshold
      - signal_quality: summary of signal quality across session
    """
    # --- By assistance state ---
    on_mwl = [r.mwl_smoothed for r in audit_rows if r.assistance_on]
    off_mwl = [r.mwl_smoothed for r in audit_rows if not r.assistance_on]

    by_state = {
        "assist_ON": _stats(on_mwl),
        "assist_OFF": _stats(off_mwl),
    }

    # --- By workload level ---
    by_level: dict[str, dict] = {}
    for block in blocks:
        block_rows = rows_by_block.get(block.name, [])
        vals = [r.mwl_smoothed for r in block_rows]
        by_level.setdefault(block.level, []).extend(vals)

    level_stats: dict[str, dict] = {}
    for level, vals in by_level.items():
        level_stats[level] = _stats(vals)

    # --- Time above threshold ---
    n_above = sum(1 for r in audit_rows if r.mwl_smoothed >= r.threshold)
    frac_above = n_above / len(audit_rows) if audit_rows else 0.0

    # --- Signal quality ---
    sq_vals = [r.signal_quality for r in audit_rows]

    return {
        "by_assistance_state": by_state,
        "by_level": level_stats,
        "time_above_threshold": {
            "fraction": float(frac_above),
            "n_above": n_above,
            "n_total": len(audit_rows),
        },
        "signal_quality": _stats(sq_vals),
        "session_duration_s": float(
            audit_rows[-1].scenario_time_s - audit_rows[0].scenario_time_s
        ) if len(audit_rows) >= 2 else 0.0,
    }


# ---------------------------------------------------------------------------
# Performance by assistance state (subtask 4)
# ---------------------------------------------------------------------------

def _find_assist_windows(
    audit_rows: list[AuditRow],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split the session timeline into assist-ON and assist-OFF windows.

    Returns (on_windows, off_windows) where each is a list of
    (start_sec, end_sec) tuples.
    """
    if not audit_rows:
        return [], []

    on_windows: list[tuple[float, float]] = []
    off_windows: list[tuple[float, float]] = []

    seg_start = audit_rows[0].scenario_time_s
    current_on = audit_rows[0].assistance_on

    for row in audit_rows[1:]:
        if row.assistance_on != current_on:
            window = (seg_start, row.scenario_time_s)
            if current_on:
                on_windows.append(window)
            else:
                off_windows.append(window)
            seg_start = row.scenario_time_s
            current_on = row.assistance_on

    # Close final segment
    final_t = audit_rows[-1].scenario_time_s
    if final_t > seg_start:
        window = (seg_start, final_t)
        if current_on:
            on_windows.append(window)
        else:
            off_windows.append(window)

    return on_windows, off_windows


def compute_performance_by_state(
    session_csv: Path,
    audit_rows: list[AuditRow],
) -> dict:
    """Compare task performance during assist-ON vs assist-OFF periods.

    Uses the existing OpenMATB performance parser to extract derived KPIs
    (tracking RMSE, sysmon accuracy, comms accuracy) within each time window,
    then aggregates across all ON windows vs all OFF windows.
    """
    from performance.summarise_openmatb_performance import (
        _collect_performance_rows,
        _compute_derived_kpis,
    )

    on_windows, off_windows = _find_assist_windows(audit_rows)

    def _aggregate_kpis(
        windows: list[tuple[float, float]],
    ) -> dict:
        """Collect performance rows across multiple windows, compute KPIs."""
        merged: dict[str, dict[str, list[str]]] = {}
        for start, end in windows:
            by_mod = _collect_performance_rows(session_csv, window=(start, end))
            for mod, metrics in by_mod.items():
                if mod not in merged:
                    merged[mod] = {}
                for metric, vals in metrics.items():
                    merged[mod].setdefault(metric, []).extend(vals)
        return _compute_derived_kpis(merged)

    on_kpis = _aggregate_kpis(on_windows) if on_windows else {}
    off_kpis = _aggregate_kpis(off_windows) if off_windows else {}

    # Total time in each state
    on_total = sum(e - s for s, e in on_windows)
    off_total = sum(e - s for s, e in off_windows)

    return {
        "assist_ON": {
            "n_windows": len(on_windows),
            "total_time_s": float(on_total),
            "kpis": on_kpis,
        },
        "assist_OFF": {
            "n_windows": len(off_windows),
            "total_time_s": float(off_total),
            "kpis": off_kpis,
        },
    }


# ---------------------------------------------------------------------------
# CLI (subtask 5 will add summary output)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse an MWL-driven adaptation session.",
    )
    parser.add_argument("--audit", type=Path, required=True,
                        help="Path to adaptation_audit.csv")
    parser.add_argument("--session", type=Path, required=True,
                        help="Path to OpenMATB session CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: print to stdout)")
    args = parser.parse_args()

    # --- Load ---
    print("Loading adaptation audit log...")
    audit_rows = load_audit_csv(args.audit)
    print(f"  {len(audit_rows)} decision ticks "
          f"({audit_rows[0].scenario_time_s:.1f}s – "
          f"{audit_rows[-1].scenario_time_s:.1f}s)")

    print("Loading OpenMATB session blocks...")
    blocks = load_session_blocks(args.session)
    if not blocks:
        print("  WARNING: No workload block markers found in session CSV.")
    else:
        level_counts = {}
        for b in blocks:
            level_counts[b.level] = level_counts.get(b.level, 0) + 1
        print(f"  {len(blocks)} blocks: {level_counts}")

    print("Assigning audit rows to blocks...")
    rows_by_block = assign_rows_to_blocks(audit_rows, blocks)
    n_assigned = sum(len(v) for k, v in rows_by_block.items() if k != "__unassigned__")
    n_unassigned = len(rows_by_block.get("__unassigned__", []))
    print(f"  {n_assigned} assigned, {n_unassigned} unassigned")

    # --- Switching dynamics ---
    print("\nComputing switching dynamics...")
    dynamics = compute_switching_dynamics(blocks, rows_by_block)
    ov = dynamics["overall"]
    print(f"  Total toggles ON:  {ov['total_toggles_on']}")
    print(f"  Total toggles OFF: {ov['total_toggles_off']}")
    print(f"  Correct detections (HIGH):      {ov['correct_detections_HIGH']}")
    print(f"  Ambiguous triggers (MODERATE):  {ov['ambiguous_triggers_MODERATE']}")
    print(f"  False alarms (LOW):             {ov['false_alarms_LOW']}")
    if "episode_duration_mean_s" in ov:
        print(f"  Episode duration: {ov['episode_duration_mean_s']:.1f}s "
              f"+- {ov['episode_duration_std_s']:.1f}s")

    high_agg = dynamics["by_level"].get("HIGH", {})
    if "detection_latency_mean_s" in high_agg:
        print(f"  Detection latency (HIGH blocks): "
              f"{high_agg['detection_latency_mean_s']:.1f}s "
              f"+- {high_agg['detection_latency_std_s']:.1f}s")

    # --- MWL trajectory ---
    print("\nComputing MWL trajectory statistics...")
    trajectory = compute_mwl_trajectory(audit_rows, blocks, rows_by_block)
    ta = trajectory["time_above_threshold"]
    print(f"  Time above threshold: {ta['fraction']:.1%} "
          f"({ta['n_above']}/{ta['n_total']} ticks)")
    for state_key in ("assist_ON", "assist_OFF"):
        st = trajectory["by_assistance_state"].get(state_key, {})
        if st:
            print(f"  MWL when {state_key}: "
                  f"{st['mean']:.3f} +- {st['std']:.3f}  (n={st['n']})")
    for level in ("HIGH", "MODERATE", "LOW"):
        st = trajectory["by_level"].get(level, {})
        if st:
            print(f"  MWL during {level}: "
                  f"{st['mean']:.3f} +- {st['std']:.3f}  (n={st['n']})")
    sq = trajectory["signal_quality"]
    if sq:
        print(f"  Signal quality: {sq['mean']:.3f} +- {sq['std']:.3f} "
              f"(min={sq['min']:.3f})")

    # --- Performance by assistance state ---
    print("\nComputing performance by assistance state...")
    perf_by_state = compute_performance_by_state(args.session, audit_rows)
    for state_key in ("assist_ON", "assist_OFF"):
        st = perf_by_state[state_key]
        kpis = st["kpis"]
        print(f"  {state_key}: {st['n_windows']} windows, "
              f"{st['total_time_s']:.1f}s total")
        if "tracking" in kpis:
            rmse = kpis["tracking"].get("center_deviation_rmse")
            if rmse is not None:
                print(f"    Tracking RMSE: {rmse:.4f}")
        if "sysmon" in kpis:
            acc = kpis["sysmon"].get("accuracy")
            if acc is not None:
                print(f"    SysMon accuracy: {acc:.1%}")
        if "communications" in kpis:
            acc = kpis["communications"].get("accuracy")
            if acc is not None:
                print(f"    Comms accuracy: {acc:.1%}")

    # --- Assemble results ---
    results = {
        "metadata": {
            "audit_path": str(args.audit),
            "session_path": str(args.session),
            "n_audit_rows": len(audit_rows),
            "n_blocks": len(blocks),
            "scenario_time_range_s": [
                audit_rows[0].scenario_time_s,
                audit_rows[-1].scenario_time_s,
            ],
        },
        "switching_dynamics": dynamics,
        "mwl_trajectory": trajectory,
        "performance_by_state": perf_by_state,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nResults written to {args.out}")
    else:
        print("\n--- Full JSON ---")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
