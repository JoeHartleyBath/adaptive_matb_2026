"""event_generators.py

Online Poisson-process event generators for MATB tasks.

Each generator models the next event time as drawn from an exponential
distribution with a rate parameter (Hz) that can be updated live as the
difficulty scalar changes.  No pre-computed timestamps are stored.

Design note: this module has ZERO vendor/OpenMATB imports so it can be
tested headlessly and imported outside the bootstrap subprocess.  The
AdaptationScheduler is responsible for converting ScheduledEvent objects
into core.event.Event objects before appending them to the scheduler queue.

Usage
-----
    gen = PoissonEventGenerator(
        plugin='communications',
        command=['radioprompt', 'own'],
        initial_rate_hz=0.02,
        seed=42,
    )
    gen.begin(start_t=0.0)

    # Inside pyglet update loop:
    while gen.ready(scenario_time):
        evt = gen.pop(scenario_time)
        # ... convert evt to core.event.Event and append
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# ScheduledEvent – vendor-free event representation
# ---------------------------------------------------------------------------

@dataclass
class ScheduledEvent:
    """Minimal event descriptor; converted to core.event.Event by AdaptationScheduler."""
    time_sec: float
    plugin: str
    command: List


# ---------------------------------------------------------------------------
# PoissonEventGenerator
# ---------------------------------------------------------------------------

class PoissonEventGenerator:
    """Homogeneous Poisson process event scheduler.

    Samples inter-event intervals from Exp(1/rate_hz).  Calling
    ``update_rate()`` changes the rate for subsequent intervals without
    resetting the currently pending ``_next_t``, which avoids artificial
    event bursts on difficulty transitions.

    Parameters
    ----------
    plugin:
        OpenMATB plugin alias (e.g. ``'communications'``).
    command:
        Two-element list passed verbatim to the plugin command, e.g.
        ``['radioprompt', 'own']`` or ``['lights-1-failure', True]``.
    initial_rate_hz:
        Starting event rate.  Must be > 0.
    seed:
        Seed for this generator's private ``random.Random`` instance.
        Pass a derived value (e.g. ``base_seed + channel_index``) so that
        each channel is independently reproducible.
    follow_up:
        Optional recovery event emitted a fixed delay after the primary
        event.  Used for pump failures (trigger + off 10 s later).
        Format: (delay_sec, plugin, command).
    """

    def __init__(
        self,
        plugin: str,
        command: List,
        initial_rate_hz: float,
        seed: Optional[int] = None,
        follow_up: Optional[Tuple[float, str, List]] = None,
    ) -> None:
        if initial_rate_hz <= 0:
            raise ValueError(f"initial_rate_hz must be > 0; got {initial_rate_hz}")

        self.plugin = plugin
        self.command = command
        self.follow_up = follow_up

        self._rate_hz: float = initial_rate_hz
        self._rng: random.Random = random.Random(seed)
        self._next_t: Optional[float] = None   # set by begin()
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin(self, start_t: float = 0.0) -> None:
        """Initialise the generator at scenario time *start_t*.

        Must be called once before ``ready()`` / ``pop()``.
        """
        self._next_t = start_t + self._sample_interval()
        self._started = True

    # ------------------------------------------------------------------
    # Runtime interface
    # ------------------------------------------------------------------

    def update_rate(self, new_rate_hz: float) -> None:
        """Update the event rate for future intervals.

        The currently pending ``_next_t`` is NOT altered to avoid creating
        artificial event bursts when difficulty changes.  The new rate
        takes effect from the next sampled interval onward.
        """
        if new_rate_hz <= 0:
            raise ValueError(f"new_rate_hz must be > 0; got {new_rate_hz}")
        self._rate_hz = new_rate_hz

    def ready(self, scenario_time: float) -> bool:
        """Return True if an event is due at *scenario_time*."""
        if not self._started:
            return False
        assert self._next_t is not None
        return scenario_time >= self._next_t

    def pop(self, scenario_time: float) -> List[ScheduledEvent]:
        """Consume a pending event and advance the schedule.

        Returns a list containing the primary event and, if a follow-up
        was configured, the follow-up event as well.

        Raises
        ------
        RuntimeError
            If called before ``begin()`` or when ``ready()`` is False.
        """
        if not self._started:
            raise RuntimeError("PoissonEventGenerator.begin() has not been called")
        if not self.ready(scenario_time):
            raise RuntimeError(
                f"pop() called at t={scenario_time:.2f} but next event not due "
                f"until t={self._next_t:.2f}"
            )

        fire_t = self._next_t
        self._next_t = fire_t + self._sample_interval()  # type: ignore[operator]

        events: List[ScheduledEvent] = [
            ScheduledEvent(time_sec=fire_t, plugin=self.plugin, command=list(self.command))
        ]

        if self.follow_up is not None:
            delay_sec, fu_plugin, fu_command = self.follow_up
            events.append(
                ScheduledEvent(
                    time_sec=fire_t + delay_sec,
                    plugin=fu_plugin,
                    command=list(fu_command),
                )
            )

        return events

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    @property
    def next_t(self) -> Optional[float]:
        """Scenario time of the next pending event; None if not started."""
        return self._next_t

    def __repr__(self) -> str:
        return (
            f"PoissonEventGenerator("
            f"plugin={self.plugin!r}, command={self.command}, "
            f"rate={self._rate_hz:.4f} Hz, next_t={self._next_t})"
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sample_interval(self) -> float:
        """Sample an exponentially-distributed inter-event interval."""
        return self._rng.expovariate(self._rate_hz)


# ---------------------------------------------------------------------------
# Factory: build the standard set of generators for a full MATB session
# ---------------------------------------------------------------------------

def build_standard_generators(
    initial_comms_rate_hz: float,
    initial_sysmon_light_rate_hz: float,
    initial_sysmon_scale_rate_hz: float,
    initial_pump_rate_hz: float,
    base_seed: int,
    pump_failure_duration_sec: float = 10.0,
) -> dict[str, PoissonEventGenerator]:
    """Construct one generator per event-channel for a full MATB session.

    Channel labels match the keys used in AdaptationScheduler._generators.

    Parameters
    ----------
    base_seed:
        Each channel receives a deterministic seed derived from this value
        so they are independently reproducible while the whole session is
        reproduced by a single seed.
    pump_failure_duration_sec:
        How long after a pump-failure event the recovery ('off') event fires.
        Default 10 s matches generate_pilot_scenarios.py.

    Returns
    -------
    dict mapping channel label → PoissonEventGenerator
    """
    # Seed derivation: sequential offsets from base_seed (simple, auditable)
    _s = base_seed

    def _next_seed() -> int:
        nonlocal _s
        _s += 1
        return _s

    gens: dict[str, PoissonEventGenerator] = {}

    # Communications
    gens["comms_own"] = PoissonEventGenerator(
        plugin="communications",
        command=["radioprompt", "own"],
        initial_rate_hz=initial_comms_rate_hz * 0.50,   # 50 % own (COMMUNICATIONS_TARGET_RATIO)
        seed=_next_seed(),
    )
    gens["comms_other"] = PoissonEventGenerator(
        plugin="communications",
        command=["radioprompt", "other"],
        initial_rate_hz=initial_comms_rate_hz * 0.50,
        seed=_next_seed(),
    )

    # SysMon lights (2 channels)
    # NOTE: must use Python bool True, NOT the string "True".  Injected events bypass
    # Scenario.check_events() type-conversion, so set_parameter receives the value as-is
    # and sysmon checks `gauge['failure'] == True` (strict equality).
    for light_id in ("1", "2"):
        key = f"sysmon_light_{light_id}"
        gens[key] = PoissonEventGenerator(
            plugin="sysmon",
            command=[f"lights-{light_id}-failure", True],
            initial_rate_hz=initial_sysmon_light_rate_hz / 2,
            seed=_next_seed(),
        )

    # SysMon scales (4 channels)
    for scale_id in ("1", "2", "3", "4"):
        key = f"sysmon_scale_{scale_id}"
        gens[key] = PoissonEventGenerator(
            plugin="sysmon",
            command=[f"scales-{scale_id}-failure", True],
            initial_rate_hz=initial_sysmon_scale_rate_hz / 4,
            seed=_next_seed(),
        )

    # ResMan pump failures (pumps 1–8; rate distributed equally)
    pump_ids = [str(i) for i in range(1, 9)]
    for pump_id in pump_ids:
        key = f"resman_pump_{pump_id}"
        gens[key] = PoissonEventGenerator(
            plugin="resman",
            command=[f"pump-{pump_id}-state", "failure"],
            initial_rate_hz=initial_pump_rate_hz / len(pump_ids),
            seed=_next_seed(),
            follow_up=(
                pump_failure_duration_sec,
                "resman",
                [f"pump-{pump_id}-state", "off"],
            ),
        )

    return gens
