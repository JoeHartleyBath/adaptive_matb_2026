"""debug_staircase.py

Quick headless debugger for the online staircase calibration.

Simulates a session with configurable score patterns and prints a rich
terminal trace so you can verify the staircase behaves as expected
without running OpenMATB.

Usage
-----
    python scripts/debug_staircase.py                      # defaults
    python scripts/debug_staircase.py --score perfect      # score=1.0 → d must rise
    python scripts/debug_staircase.py --score poor         # score=0.0 → d must fall
    python scripts/debug_staircase.py --score converge     # noisy around target
    python scripts/debug_staircase.py --score alternating  # oscillates → tests reversal count
    python scripts/debug_staircase.py --score custom --custom-value 0.85
    python scripts/debug_staircase.py --seed 99 --duration 300 --window 30
    python scripts/debug_staircase.py --help

Output
------
  - A live event log (each step printed as it fires)
  - An ASCII timeline of d(t)
  - A final summary table
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adaptation.difficulty_state import DifficultyState
from adaptation.staircase_controller import StaircaseController
from adaptation.event_generators import build_standard_generators


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug the staircase calibration headlessly.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--score",
        choices=["perfect", "poor", "converge", "alternating", "noisy", "custom"],
        default="noisy",
        help=(
            "Score pattern fed to the staircase. "
            "'converge' = noisy around target; "
            "'alternating' = blocks of high/low to force reversals."
        ),
    )
    p.add_argument(
        "--custom-value", type=float, default=0.80, metavar="V",
        help="Constant score value used when --score=custom.",
    )
    p.add_argument("--target", type=float, default=0.70, help="Staircase target score.")
    p.add_argument("--tolerance", type=float, default=0.05, help="Dead-band tolerance.")
    p.add_argument("--window", type=float, default=45.0, help="Evaluation window (s).")
    p.add_argument(
        "--step-schedule", default="0.2,0.1,0.05",
        help="Comma-separated graduated step sizes, coarse first (e.g. 0.2,0.1,0.05).",
    )
    p.add_argument(
        "--stable-ticks", type=int, default=3,
        help="Consecutive no-step ticks at finest step required for convergence.",
    )
    p.add_argument("--cooldown", type=float, default=20.0, help="Cooldown between steps (s).")
    p.add_argument("--d-init", type=float, default=0.50, help="Starting difficulty.")
    p.add_argument("--d-min", type=float, default=0.0, help="Lower difficulty bound.")
    p.add_argument("--d-max", type=float, default=1.0, help="Upper difficulty bound.")
    p.add_argument("--duration", type=float, default=300.0, help="Simulated session length (s).")
    p.add_argument("--sample-dt", type=float, default=2.0,
                   help="Interval between performance samples (s) — simulates tracking update rate.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for noisy patterns.")
    p.add_argument("--ascii-width", type=int, default=80,
                   help="Character width of the ASCII timeline.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-step event log; show summary only.")
    p.add_argument("--show-params", action="store_true",
                   help="Print derived task params alongside each difficulty step.")
    p.add_argument("--generators", action="store_true",
                   help="Also simulate Poisson event generators and print first-event times.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Score generators
# ---------------------------------------------------------------------------

def _make_score_fn(args: argparse.Namespace):
    rng = random.Random(args.seed)
    alternating_state = {"direction": "high", "counter": 0, "block_len": 60}

    def fn(t: float) -> float:
        if args.score == "perfect":
            return 1.0
        if args.score == "poor":
            return 0.0
        if args.score == "custom":
            return max(0.0, min(1.0, args.custom_value))
        if args.score == "converge":
            # Noisy near the target; should produce few steps
            return max(0.0, min(1.0, rng.gauss(args.target, 0.06)))
        if args.score == "noisy":
            # Noisy, slightly above target → should drift d upward
            return max(0.0, min(1.0, rng.gauss(args.target + 0.10, 0.12)))
        if args.score == "alternating":
            st = alternating_state
            st["counter"] += 1
            if st["counter"] >= st["block_len"] / args.sample_dt:
                st["counter"] = 0
                st["direction"] = "low" if st["direction"] == "high" else "high"
            if st["direction"] == "high":
                return max(0.0, min(1.0, rng.gauss(0.90, 0.05)))
            else:
                return max(0.0, min(1.0, rng.gauss(0.40, 0.05)))
        return args.target

    return fn


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class _StepRecord:
    t: float
    d_before: float
    d_after: float
    delta: float
    window_mean: float
    reversal_count: int
    step_up: float
    step_down: float
    schedule_idx: int


def run_simulation(args: argparse.Namespace) -> Tuple[List[Tuple[float, float]], List[_StepRecord]]:
    """Return (trajectory, steps) where trajectory is list of (t, d)."""
    state = DifficultyState(
        d_init=args.d_init,
        d_min=args.d_min,
        d_max=args.d_max,
        seed=args.seed,
    )
    step_schedule = tuple(float(x) for x in args.step_schedule.split(",") if x.strip())
    ctrl = StaircaseController(
        target_score=args.target,
        tolerance=args.tolerance,
        window_sec=args.window,
        min_samples=3,
        step_schedule=step_schedule,
        cooldown_sec=args.cooldown,
        stable_ticks_required=args.stable_ticks,
    )
    score_fn = _make_score_fn(args)

    trajectory: List[Tuple[float, float]] = [(0.0, state.d)]
    steps: List[_StepRecord] = []

    t = 0.0
    while t <= args.duration:
        t = round(t + args.sample_dt, 6)
        score = score_fn(t)
        ctrl.push_performance(t, score)
        delta = ctrl.tick(t)
        if ctrl.converged:
            print(f"\n>>> CONVERGED at t={t:.1f}s  d={state.d:.3f}  (block would end here)")            
            trajectory.append((t, state.d))
            break
        if delta is not None:
            d_before = state.d
            state.update(state.d + delta)
            steps.append(
                _StepRecord(
                    t=t,
                    d_before=d_before,
                    d_after=state.d,
                    delta=delta,
                    window_mean=ctrl.window_mean or 0.0,
                    reversal_count=ctrl.reversal_count,
                    step_up=ctrl.step_up,
                    step_down=ctrl.step_down,
                    schedule_idx=ctrl._schedule_idx,
                )
            )
        trajectory.append((t, state.d))

    return trajectory, steps


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_ESC = "\033["
_RESET = f"{_ESC}0m"
_BOLD = f"{_ESC}1m"
_GREEN = f"{_ESC}32m"
_RED = f"{_ESC}31m"
_YELLOW = f"{_ESC}33m"
_CYAN = f"{_ESC}36m"
_DIM = f"{_ESC}2m"


def _colour(text: str, code: str) -> str:
    # Disable colour when not a TTY (e.g. piped to file)
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{_RESET}"


def print_event_log(steps: List[_StepRecord], args: argparse.Namespace) -> None:
    if not steps:
        print(_colour("  (no staircase steps fired)", _DIM))
        return
    prev_rev = 0
    for s in steps:
        direction = "▲ UP  " if s.delta > 0 else "▼ DOWN"
        colour = _GREEN if s.delta > 0 else _RED
        reversal_marker = ""
        if s.reversal_count > prev_rev:
            reversal_marker = _colour(f"  ↩ reversal #{s.reversal_count}", _YELLOW)
            prev_rev = s.reversal_count
        line = (
            f"  t={s.t:6.1f}s  {_colour(direction, colour)}"
            f"  d: {s.d_before:.3f} → {_colour(f'{s.d_after:.3f}', colour)}"
            f"  Δ={s.delta:+.4f}"
            f"  window_mean={s.window_mean:.3f}"
            f"  step={s.step_up:.4f}"
            f"{reversal_marker}"
        )
        print(line)
        if args.show_params:
            from adaptation.difficulty_state import DifficultyState as _DS
            _st = _DS(d_init=s.d_after)
            p = _st.params
            print(
                f"           "
                f"track_update={p.track_update_ms:.0f}ms  "
                f"joystick_force={p.track_joystick_force:.2f}  "
                f"resman_leak={p.resman_loss_a_per_min}ml/min  "
                f"comms_rate={p.comms_rate_hz:.4f}Hz"
            )


def print_summary(
    trajectory: List[Tuple[float, float]],
    steps: List[_StepRecord],
    args: argparse.Namespace,
) -> None:
    d_values = [d for _, d in trajectory]
    d_final = d_values[-1]
    n_up = sum(1 for s in steps if s.delta > 0)
    n_down = sum(1 for s in steps if s.delta < 0)
    reversals = steps[-1].reversal_count if steps else 0
    step_final = steps[-1].step_up if steps else float(args.step_schedule.split(",")[0])
    schedule_str = args.step_schedule

    print()
    print(_colour("━" * 60, _BOLD))
    print(_colour("  SUMMARY", _BOLD))
    print(_colour("━" * 60, _BOLD))
    rows = [
        ("Score pattern",        args.score),
        ("Seed",                  str(args.seed)),
        ("Duration",              f"{args.duration:.0f}s"),
        ("Window",                f"{args.window:.0f}s"),
        ("Target score",          f"{args.target:.2f}  ±{args.tolerance:.2f}"),
        ("Step schedule",         schedule_str),
        ("Stable ticks req.",     str(args.stable_ticks)),
        ("d_init → d_final",      f"{args.d_init:.3f} → {d_final:.3f}"),
        ("d min / max seen",      f"{min(d_values):.3f} / {max(d_values):.3f}"),
        ("Steps total",           f"{len(steps)}  (↑{n_up}  ↓{n_down})"),
        ("Reversals",             str(reversals)),
        ("Step size at end",      f"{step_final:.4f}"),
    ]
    for label, value in rows:
        print(f"  {label:<24}  {value}")

    # Expectation checks
    print()
    print(_colour("  CHECKS", _BOLD))
    checks = []
    initial_step = float(args.step_schedule.split(",")[0].strip())
    final_schedule_step = float(args.step_schedule.split(",")[-1].strip())

    # Bounds
    bound_ok = all(args.d_min <= d <= args.d_max for _, d in trajectory)
    checks.append(("d stays within [d_min, d_max]", bound_ok))

    # Direction
    if args.score == "perfect":
        checks.append(("d increased overall (score=1.0 → harder)",  d_final > args.d_init))
    elif args.score == "poor":
        checks.append(("d decreased overall (score=0.0 → easier)",  d_final < args.d_init))
    elif args.score in ("noisy",):
        checks.append(("At least one step fired",                    len(steps) > 0))
    elif args.score == "converge":
        checks.append(("Fewer than 10 steps (near-target noise)",   len(steps) < 10))
    elif args.score == "alternating":
        checks.append(("At least 2 reversals detected",             reversals >= 2))
        checks.append(("Step size reduced from initial",            step_final < initial_step))

    for label, ok in checks:
        mark = _colour("  ✓", _GREEN) if ok else _colour("  ✗", _RED)
        print(f"{mark}  {label}" + ("" if ok else _colour("  ← UNEXPECTED", _RED)))


def print_ascii_timeline(
    trajectory: List[Tuple[float, float]],
    steps: List[_StepRecord],
    args: argparse.Namespace,
) -> None:
    """Render a character-art plot of d(t) over the session."""
    width = args.ascii_width
    height = 16
    d_min_plot = 0.0
    d_max_plot = 1.0
    t_max = args.duration

    # Build a grid (height rows × width cols)
    grid = [[" "] * width for _ in range(height)]

    def _col(t: float) -> int:
        return min(width - 1, int((t / t_max) * (width - 1)))

    def _row(d: float) -> int:
        pct = (d - d_min_plot) / (d_max_plot - d_min_plot)
        return max(0, min(height - 1, height - 1 - int(pct * (height - 1))))

    # Target band
    for col in range(width):
        for d_band in [args.target + args.tolerance, args.target - args.tolerance, args.target]:
            r = _row(d_band)
            if grid[r][col] == " ":
                grid[r][col] = "·" if d_band == args.target else "—"

    # Trajectory dots
    prev_col = -1
    for t_, d_ in trajectory:
        c = _col(t_)
        r = _row(d_)
        if c != prev_col:
            grid[r][c] = "█" if any(abs(s.t - t_) < args.sample_dt for s in steps) else "▪"
            prev_col = c

    # Step markers (vertical bar)
    for s in steps:
        c = _col(s.t)
        for r in range(height):
            if grid[r][c] == " " or grid[r][c] in ("·", "—"):
                grid[r][c] = "│"
        grid[_row(s.d_after)][c] = "◆"

    # Print
    print()
    print(_colour("  d(t) over session", _BOLD))
    print(f"  {'1.0':>5} ┐")
    for i, row in enumerate(grid):
        d_label = 1.0 - (i / (height - 1))
        marker = "→" if abs(d_label - args.target) < (0.5 / height) else " "
        print(f"  {d_label:4.2f} {marker}│ {''.join(row)}")
    print(f"  {'0.0':>5} ┘")
    print(f"        └{'─' * width}")
    print(f"          0s{' ' * (width // 2 - 4)}t → {t_max:.0f}s")
    print()
    print(_colour(
        f"  ▪=trajectory  ◆=step  │=step column  ·=target  —=tolerance band",
        _DIM,
    ))


# ---------------------------------------------------------------------------
# Generator probe (optional)
# ---------------------------------------------------------------------------

def print_generator_probe(args: argparse.Namespace) -> None:
    state = DifficultyState(d_init=args.d_init, seed=args.seed)
    p = state.params
    gens = build_standard_generators(
        initial_comms_rate_hz=p.comms_rate_hz,
        initial_sysmon_light_rate_hz=p.sysmon_light_rate_hz,
        initial_sysmon_scale_rate_hz=p.sysmon_scale_rate_hz,
        initial_pump_rate_hz=p.resman_pump_rate_hz,
        base_seed=args.seed,
    )
    for name, gen in gens.items():
        gen.begin(0.0)

    print()
    print(_colour("  GENERATOR FIRST-EVENT TIMES  (d_init)", _BOLD))
    print(f"  {'channel':<25}  {'rate (Hz)':>10}  {'mean IEI (s)':>13}  {'first event (s)':>16}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*13}  {'─'*16}")
    for name, gen in gens.items():
        rate = gen.rate_hz
        mean_iei = 1.0 / rate if rate > 0 else float("inf")
        first = gen.next_t if gen.next_t is not None else float("nan")
        print(f"  {name:<25}  {rate:>10.5f}  {mean_iei:>13.1f}  {first:>16.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print()
    print(_colour("━" * 60, _BOLD))
    print(_colour("  STAIRCASE DEBUG SIMULATION", _BOLD))
    print(_colour("━" * 60, _BOLD))
    print(f"  score={args.score}  seed={args.seed}  duration={args.duration:.0f}s")
    print(f"  target={args.target}±{args.tolerance}  window={args.window}s")
    print(f"  step_schedule={args.step_schedule}  cooldown={args.cooldown}s  d_init={args.d_init}")
    print()

    trajectory, steps = run_simulation(args)

    if not args.quiet:
        print(_colour("  STEP EVENT LOG", _BOLD))
        print_event_log(steps, args)

    print_ascii_timeline(trajectory, steps, args)
    print_summary(trajectory, steps, args)

    if args.generators:
        print_generator_probe(args)

    print()
    # Exit non-zero if any check failed (useful in CI)
    checks_all_ok = True
    d_values = [d for _, d in trajectory]
    if not all(args.d_min <= d <= args.d_max for _, d in trajectory):
        checks_all_ok = False
    sys.exit(0 if checks_all_ok else 1)


if __name__ == "__main__":
    main()
