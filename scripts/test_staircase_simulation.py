"""test_staircase_simulation.py

Headless validation of the online staircase calibration components.
No pyglet, no OpenMATB vendor imports required.

Run with:
    python scripts/test_staircase_simulation.py

Or, if pytest is available:
    pytest scripts/test_staircase_simulation.py -v

Tests
-----
1. DifficultyState – parameter ranges and monotonicity
2. Staircase monotone – perfect scores drive d upward; d never exceeds bounds
3. Staircase reversal – step size halves after n reversals
4. Staircase reproducibility – identical seeds → identical trajectories
5. Generator rate-response – mean IEI shortens after rate increase
6. Generator reproducibility – same seed → same event sequence
"""

from __future__ import annotations

import sys
import os

# Make src/python importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src", "python"))

import math
import random
import unittest

from adaptation.difficulty_state import DifficultyState, _resman_leak
from adaptation.staircase_controller import StaircaseController
from adaptation.event_generators import PoissonEventGenerator, build_standard_generators


# ---------------------------------------------------------------------------
# 1. DifficultyState
# ---------------------------------------------------------------------------

class TestDifficultyState(unittest.TestCase):

    def test_parameter_ranges_at_extremes(self):
        lo = DifficultyState(d_init=0.0)
        hi = DifficultyState(d_init=1.0)

        # Tracking: lower update_ms and lower joystick_force at high difficulty
        self.assertGreater(lo.params.track_update_ms, hi.params.track_update_ms)
        self.assertGreater(lo.params.track_joystick_force, hi.params.track_joystick_force)

        # Resman: higher leak at higher difficulty
        self.assertLess(lo.params.resman_loss_a_per_min, hi.params.resman_loss_a_per_min)

        # Event rates: higher at higher difficulty
        self.assertLess(lo.params.comms_rate_hz, hi.params.comms_rate_hz)
        self.assertLess(lo.params.sysmon_light_rate_hz, hi.params.sysmon_light_rate_hz)
        self.assertLess(lo.params.resman_pump_rate_hz, hi.params.resman_pump_rate_hz)

    def test_all_rates_strictly_positive(self):
        for d_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            state = DifficultyState(d_init=d_val)
            self.assertGreater(state.params.comms_rate_hz, 0.0, f"comms_rate at d={d_val}")
            self.assertGreater(state.params.sysmon_light_rate_hz, 0.0)
            self.assertGreater(state.params.resman_pump_rate_hz, 0.0)

    def test_d_prev_tracks_previous_value(self):
        state = DifficultyState(d_init=0.5)
        state.update(0.7)
        self.assertAlmostEqual(state.d_prev, 0.5)
        self.assertAlmostEqual(state.d, 0.7)

    def test_clamping(self):
        state = DifficultyState(d_init=0.5, d_min=0.2, d_max=0.8)
        state.update(1.5)
        self.assertAlmostEqual(state.d, 0.8)
        state.update(-0.3)
        self.assertAlmostEqual(state.d, 0.2)

    def test_resman_leak_key_values(self):
        # Matches expected values from generate_pilot_scenarios.py
        self.assertEqual(_resman_leak(0.20), 240)   # 1200*0.20 = 240, no offset
        self.assertEqual(_resman_leak(0.55), 560)   # 1200*0.55=660, -100=560
        self.assertEqual(_resman_leak(0.95), 940)   # 1200*0.95=1140, -100-100=940

    def test_as_dict_is_serialisable(self):
        import json
        state = DifficultyState(d_init=0.6, seed=99)
        snapshot = state.as_dict()
        json_str = json.dumps(snapshot)   # raises if not serialisable
        self.assertIn("d", json_str)
        self.assertIn("track_update_ms", json_str)

    def test_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            DifficultyState(d_min=0.8, d_max=0.2)


# ---------------------------------------------------------------------------
# 2. StaircaseController – monotone response
# ---------------------------------------------------------------------------

class TestStaircaseMonotone(unittest.TestCase):

    def _run_simulation(
        self,
        *,
        score_value: float,
        n_seconds: int = 300,
        sample_interval_sec: float = 2.0,
        window_sec: float = 45.0,
        step: float = 0.05,
        target: float = 0.70,
        tolerance: float = 0.05,
        cooldown_sec: float = 20.0,
        d_init: float = 0.5,
        d_min: float = 0.0,
        d_max: float = 1.0,
    ):
        """Feed constant scores for n_seconds; return list of (t, d)."""
        state = DifficultyState(d_init=d_init, d_min=d_min, d_max=d_max)
        ctrl = StaircaseController(
            target_score=target,
            tolerance=tolerance,
            window_sec=window_sec,
            step_up=step,
            step_down=step,
            cooldown_sec=cooldown_sec,
            n_reversals_to_halve=0,   # disable halving for monotone test
        )

        trajectory = [(0.0, state.d)]
        t = 0.0
        while t <= n_seconds:
            t += sample_interval_sec
            ctrl.push_performance(t, score_value)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
            trajectory.append((t, state.d))

        return trajectory

    def test_perfect_score_drives_d_up(self):
        """Constant score=1.0 (above target 0.70) should drive d upward."""
        traj = self._run_simulation(score_value=1.0)
        d_values = [d for _, d in traj]
        # d must have increased at some point
        self.assertGreater(max(d_values), 0.5, "d never increased with perfect scores")
        # d must be monotone non-decreasing (1-up/1-down with no reversals)
        for i in range(1, len(d_values)):
            self.assertGreaterEqual(
                d_values[i], d_values[i - 1],
                f"d decreased at index {i} with perfect scores",
            )

    def test_poor_score_drives_d_down(self):
        """Constant score=0.0 (below target 0.70) should drive d downward."""
        traj = self._run_simulation(score_value=0.0)
        d_values = [d for _, d in traj]
        self.assertLess(min(d_values), 0.5, "d never decreased with zero scores")
        for i in range(1, len(d_values)):
            self.assertLessEqual(
                d_values[i], d_values[i - 1],
                f"d increased at index {i} with zero scores",
            )

    def test_d_never_exceeds_bounds(self):
        """d must remain within [d_min, d_max] regardless of scores."""
        for score in [0.0, 1.0]:
            traj = self._run_simulation(
                score_value=score, d_min=0.1, d_max=0.9, n_seconds=600
            )
            for t, d in traj:
                self.assertGreaterEqual(d, 0.1, f"d={d} below d_min at t={t}")
                self.assertLessEqual(d, 0.9, f"d={d} above d_max at t={t}")

    def test_on_target_no_change(self):
        """Score exactly at target must produce no steps."""
        state = DifficultyState(d_init=0.5)
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=45.0,
            cooldown_sec=5.0,
            n_reversals_to_halve=0,
        )
        # Fill window with target score
        for t in range(1, 200):
            ctrl.push_performance(float(t), 0.70)
            delta = ctrl.tick(float(t))
            if delta is not None:
                state.update(state.d + delta)

        self.assertAlmostEqual(state.d, 0.5, places=5, msg="d changed despite on-target score")


# ---------------------------------------------------------------------------
# 3. Staircase reversal and step-size halving
# ---------------------------------------------------------------------------

class TestStaircaseReversals(unittest.TestCase):

    def test_step_size_halves_after_n_reversals(self):
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=30.0,
            step_up=0.10,
            step_down=0.10,
            cooldown_sec=5.0,
            n_reversals_to_halve=2,
            min_step=0.005,
        )
        state = DifficultyState(d_init=0.5)
        original_step = ctrl.step_up

        t = 0.0
        dt = 1.0
        direction = "up"   # feed alternating high/low blocks

        # simulate long enough to force 4+ reversals
        while ctrl.reversal_count < 4 and t < 2000:
            score = 1.0 if direction == "up" else 0.0
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
                # flip score direction
                direction = "down" if delta > 0 else "up"
            t += dt

        self.assertGreaterEqual(ctrl.reversal_count, 4, "Not enough reversals generated")
        # Step size should have halved at least once (at rev 2 and again at rev 4)
        self.assertLess(ctrl.step_up, original_step, "Step size was not reduced")

    def test_step_size_floor(self):
        """Step size never falls below min_step."""
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=20.0,
            step_up=0.20,
            step_down=0.20,
            cooldown_sec=5.0,
            n_reversals_to_halve=1,   # halve on every reversal
            min_step=0.01,
        )
        state = DifficultyState(d_init=0.5)
        t = 0.0
        direction = "up"
        for _ in range(500):
            score = 1.0 if direction == "up" else 0.0
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
                direction = "down" if delta > 0 else "up"
            t += 1.0
            self.assertGreaterEqual(ctrl.step_up, 0.01, "step_up fell below min_step")
            self.assertGreaterEqual(ctrl.step_down, 0.01, "step_down fell below min_step")


# ---------------------------------------------------------------------------
# 4. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility(unittest.TestCase):

    def _run_deterministic_simulation(self, *, n_seconds: int, rng_seed: int):
        """Use a seeded RNG to generate noisy scores; return d-trajectory list."""
        rng = random.Random(rng_seed)
        state = DifficultyState(d_init=0.5)
        ctrl = StaircaseController(
            target_score=0.70,
            window_sec=30.0,
            cooldown_sec=10.0,
            n_reversals_to_halve=0,
        )

        trajectory = []
        for t_int in range(1, n_seconds + 1):
            t = float(t_int)
            score = max(0.0, min(1.0, rng.gauss(0.70, 0.15)))
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
            trajectory.append(round(state.d, 8))

        return trajectory

    def test_same_seed_same_trajectory(self):
        traj_a = self._run_deterministic_simulation(n_seconds=300, rng_seed=42)
        traj_b = self._run_deterministic_simulation(n_seconds=300, rng_seed=42)
        self.assertEqual(traj_a, traj_b, "Trajectories differ with identical seed")

    def test_different_seed_different_trajectory(self):
        traj_a = self._run_deterministic_simulation(n_seconds=300, rng_seed=42)
        traj_b = self._run_deterministic_simulation(n_seconds=300, rng_seed=99)
        self.assertNotEqual(traj_a, traj_b, "Trajectories identical with different seeds")


# ---------------------------------------------------------------------------
# 5. PoissonEventGenerator – rate-response
# ---------------------------------------------------------------------------

class TestGeneratorRateResponse(unittest.TestCase):

    def _collect_intervals(
        self, gen: PoissonEventGenerator, start_t: float, end_t: float
    ):
        """Advance scenario time and collect all inter-event intervals."""
        intervals = []
        prev_t = start_t
        t = start_t
        while t <= end_t:
            if gen.ready(t):
                gen.pop(t)
                intervals.append(t - prev_t)
                prev_t = t
            t += 0.01   # 10 ms resolution
        return intervals

    def test_mean_iei_shortens_after_rate_increase(self):
        """Events should arrive more frequently after update_rate()."""
        gen = PoissonEventGenerator(
            plugin="communications",
            command=["radioprompt", "own"],
            initial_rate_hz=0.01,   # 1 event per 100 s on average
            seed=0,
        )
        gen.begin(start_t=0.0)

        intervals_before = self._collect_intervals(gen, 0.0, 500.0)
        gen.update_rate(0.20)   # 1 event per 5 s on average
        intervals_after = self._collect_intervals(gen, 500.0, 700.0)

        if len(intervals_before) < 2 or len(intervals_after) < 2:
            self.skipTest("Insufficient events sampled — increase simulation time")

        mean_before = sum(intervals_before) / len(intervals_before)
        mean_after = sum(intervals_after) / len(intervals_after)
        self.assertGreater(
            mean_before, mean_after,
            f"Mean IEI before ({mean_before:.1f}s) not > after ({mean_after:.1f}s)",
        )

    def test_ready_false_before_begin(self):
        gen = PoissonEventGenerator("sysmon", ["lights-1-failure", True], 0.05, seed=1)
        self.assertFalse(gen.ready(100.0))

    def test_pop_raises_before_begin(self):
        gen = PoissonEventGenerator("sysmon", ["lights-1-failure", True], 0.05, seed=1)
        with self.assertRaises(RuntimeError):
            gen.pop(100.0)

    def test_pop_raises_when_not_ready(self):
        gen = PoissonEventGenerator("sysmon", ["lights-1-failure", True], 0.05, seed=1)
        gen.begin(0.0)
        with self.assertRaises(RuntimeError):
            gen.pop(0.0001)   # before first event is due

    def test_follow_up_event_emitted(self):
        """Pump failure generator must emit trigger + recovery pair."""
        gen = PoissonEventGenerator(
            plugin="resman",
            command=["pump-1-state", "failure"],
            initial_rate_hz=1.0,   # fast, for testing
            seed=7,
            follow_up=(10.0, "resman", ["pump-1-state", "off"]),
        )
        gen.begin(start_t=0.0)
        # Advance until first event fires
        t = 0.0
        while not gen.ready(t):
            t += 0.01
        events = gen.pop(t)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].command, ["pump-1-state", "failure"])
        self.assertEqual(events[1].command, ["pump-1-state", "off"])
        self.assertAlmostEqual(
            events[1].time_sec - events[0].time_sec, 10.0, places=3
        )


# ---------------------------------------------------------------------------
# 6. Generator reproducibility
# ---------------------------------------------------------------------------

class TestGeneratorReproducibility(unittest.TestCase):

    def _collect_event_times(self, seed: int, n_events: int = 20) -> list:
        gen = PoissonEventGenerator(
            plugin="communications",
            command=["radioprompt", "own"],
            initial_rate_hz=0.10,
            seed=seed,
        )
        gen.begin(start_t=0.0)
        times = []
        t = 0.0
        while len(times) < n_events:
            if gen.ready(t):
                evts = gen.pop(t)
                times.append(round(evts[0].time_sec, 6))
            t += 0.001
        return times

    def test_same_seed_same_sequence(self):
        seq_a = self._collect_event_times(seed=42)
        seq_b = self._collect_event_times(seed=42)
        self.assertEqual(seq_a, seq_b)

    def test_different_seed_different_sequence(self):
        seq_a = self._collect_event_times(seed=42)
        seq_b = self._collect_event_times(seed=99)
        self.assertNotEqual(seq_a, seq_b)

    def test_build_standard_generators_deterministic(self):
        """build_standard_generators must yield reproducible event sequences."""
        def _first_event_time(seed: int) -> float:
            gens = build_standard_generators(
                initial_comms_rate_hz=0.02,
                initial_sysmon_light_rate_hz=0.02,
                initial_sysmon_scale_rate_hz=0.02,
                initial_pump_rate_hz=0.01,
                base_seed=seed,
            )
            gen = gens["comms_own"]
            gen.begin(0.0)
            t = 0.0
            while not gen.ready(t):
                t += 0.01
            return round(t, 3)

        self.assertEqual(_first_event_time(42), _first_event_time(42))
        self.assertNotEqual(_first_event_time(42), _first_event_time(43))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
