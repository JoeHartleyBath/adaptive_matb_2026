"""Simulated MWL source for desk-testing the adaptation loop without EEG.

Publishes deterministic MWL scores to an LSL outlet, matching the format
produced by ``mwl_estimator.py`` (3-channel float32: mwl_value, confidence,
signal_quality).

Modes
-----
constant   — fixed MWL value (steady-state testing)
sinusoid   — oscillating MWL (policy response testing)
block      — step function with configurable durations/values
noisy_block — block + Gaussian noise for realism
replay     — replay MWL scores from a CSV file

Usage
-----
    python -m mwl_simulated --mode block --durations 60,60,60 --values 0.2,0.8,0.2 --rate 4
    python -m mwl_simulated --mode constant --value 0.7
    python -m mwl_simulated --mode sinusoid --period 30 --amplitude 0.3 --offset 0.5
    python -m mwl_simulated --mode replay --csv results/mwl_replay.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time

import numpy as np
import pylsl

log = logging.getLogger(__name__)

_OUTLET_NAME = "MWL"
_OUTLET_TYPE = "MWL"
_DEFAULT_RATE = 4.0  # Hz — matches mwl_estimator step rate


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _gen_constant(t: float, *, value: float, **_kw) -> float:
    return value


def _gen_sinusoid(
    t: float, *, period: float, amplitude: float, offset: float, **_kw,
) -> float:
    return offset + amplitude * np.sin(2.0 * np.pi * t / period)


def _gen_block(
    t: float, *, durations: list[float], values: list[float], **_kw,
) -> float:
    elapsed = 0.0
    for dur, val in zip(durations, values):
        if t < elapsed + dur:
            return val
        elapsed += dur
    return values[-1]  # hold last value after all blocks


def _gen_noisy_block(
    t: float,
    *,
    durations: list[float],
    values: list[float],
    noise_std: float,
    rng: np.random.Generator,
    **_kw,
) -> float:
    base = _gen_block(t, durations=durations, values=values)
    return float(np.clip(base + rng.normal(0, noise_std), 0.0, 1.0))


# ---------------------------------------------------------------------------
# LSL outlet (same format as mwl_estimator)
# ---------------------------------------------------------------------------

def _create_outlet(name: str, rate: float) -> pylsl.StreamOutlet:
    info = pylsl.StreamInfo(
        name=name,
        type=_OUTLET_TYPE,
        channel_count=3,
        nominal_srate=rate,
        channel_format=pylsl.cf_float32,
        source_id="mwl_simulated",
    )
    chns = info.desc().append_child("channels")
    for label in ("mwl_value", "confidence", "signal_quality"):
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    return pylsl.StreamOutlet(info)


# ---------------------------------------------------------------------------
# Main loops
# ---------------------------------------------------------------------------

def _run_generator(
    gen_fn,
    gen_kwargs: dict,
    rate: float,
    outlet: pylsl.StreamOutlet,
    total_duration: float | None,
) -> None:
    """Push samples from a generator function until interrupted."""
    step = 1.0 / rate
    t0 = time.perf_counter()
    n = 0

    while True:
        t_elapsed = time.perf_counter() - t0
        if total_duration is not None and t_elapsed >= total_duration:
            log.info("Reached end of duration (%.1f s) — stopping.", total_duration)
            break

        mwl = float(np.clip(gen_fn(t_elapsed, **gen_kwargs), 0.0, 1.0))
        confidence = min(1.0, abs(mwl - 0.5) * 2.0)
        outlet.push_sample([mwl, confidence, 1.0])
        n += 1

        if n % int(rate * 10) == 0:
            log.info("t=%.1f s  MWL=%.3f  (n=%d)", t_elapsed, mwl, n)

        # Throttle
        next_t = t0 + n * step
        sleep_dur = next_t - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)


def _run_replay(
    csv_path: str,
    rate: float,
    outlet: pylsl.StreamOutlet,
) -> None:
    """Replay MWL values from a CSV file (column 'mwl_value' or first col)."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        log.error("CSV is empty: %s", csv_path)
        return

    col = "mwl_value" if "mwl_value" in rows[0] else list(rows[0].keys())[0]
    values = [float(r[col]) for r in rows]
    log.info("Replaying %d samples from %s (column '%s')", len(values), csv_path, col)

    step = 1.0 / rate
    t0 = time.perf_counter()

    for i, mwl in enumerate(values):
        mwl = float(np.clip(mwl, 0.0, 1.0))
        confidence = min(1.0, abs(mwl - 0.5) * 2.0)
        outlet.push_sample([mwl, confidence, 1.0])

        if (i + 1) % int(rate * 10) == 0:
            log.info("t=%.1f s  MWL=%.3f  (%d/%d)", i * step, mwl, i + 1, len(values))

        next_t = t0 + (i + 1) * step
        sleep_dur = next_t - time.perf_counter()
        if sleep_dur > 0:
            time.sleep(sleep_dur)

    log.info("Replay complete (%d samples).", len(values))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulated MWL source for desk-testing adaptation.",
    )
    p.add_argument(
        "--mode", required=True,
        choices=["constant", "sinusoid", "block", "noisy_block", "replay"],
    )
    p.add_argument("--rate", type=float, default=_DEFAULT_RATE,
                   help="Output rate in Hz (default: 4)")
    p.add_argument("--outlet-name", default=_OUTLET_NAME)

    # constant
    p.add_argument("--value", type=float, default=0.5,
                   help="MWL value for constant mode (default: 0.5)")

    # sinusoid
    p.add_argument("--period", type=float, default=30.0,
                   help="Period in seconds for sinusoid mode (default: 30)")
    p.add_argument("--amplitude", type=float, default=0.3,
                   help="Amplitude for sinusoid mode (default: 0.3)")
    p.add_argument("--offset", type=float, default=0.5,
                   help="Centre offset for sinusoid mode (default: 0.5)")

    # block / noisy_block
    p.add_argument("--durations", type=str, default="60,60,60",
                   help="Comma-separated block durations in seconds")
    p.add_argument("--values", type=str, default="0.2,0.8,0.2",
                   help="Comma-separated MWL values per block")
    p.add_argument("--noise-std", type=float, default=0.05,
                   help="Noise σ for noisy_block mode (default: 0.05)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for noisy_block (default: 42)")

    # replay
    p.add_argument("--csv", type=str, default=None,
                   help="CSV file path for replay mode")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [SIM-MWL] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args(argv)

    # Graceful shutdown (only in main thread)
    import threading
    if threading.current_thread() is threading.main_thread():
        def _handle_signal(sig, frame):
            log.info("Caught signal %s — stopping.", sig)
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_signal)

    outlet = _create_outlet(args.outlet_name, args.rate)
    log.info("Simulated MWL outlet '%s' ready (mode=%s, rate=%.1f Hz)",
             args.outlet_name, args.mode, args.rate)

    if args.mode == "replay":
        if not args.csv:
            log.error("--csv is required for replay mode.")
            sys.exit(1)
        _run_replay(args.csv, args.rate, outlet)
        return

    # Build generator function + kwargs
    durations = [float(x) for x in args.durations.split(",")]
    values = [float(x) for x in args.values.split(",")]

    generators = {
        "constant": (_gen_constant, {"value": args.value}),
        "sinusoid": (_gen_sinusoid, {
            "period": args.period,
            "amplitude": args.amplitude,
            "offset": args.offset,
        }),
        "block": (_gen_block, {
            "durations": durations,
            "values": values,
        }),
        "noisy_block": (_gen_noisy_block, {
            "durations": durations,
            "values": values,
            "noise_std": args.noise_std,
            "rng": np.random.default_rng(args.seed),
        }),
    }

    gen_fn, gen_kwargs = generators[args.mode]
    total = sum(durations) if args.mode in ("block", "noisy_block") else None

    try:
        _run_generator(gen_fn, gen_kwargs, args.rate, outlet, total)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Simulated MWL source stopped.")


if __name__ == "__main__":
    main()
