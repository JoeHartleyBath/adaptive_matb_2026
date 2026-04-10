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
3. Staircase schedule – step advances 0.2→0.1→0.05 on reversals; never below final entry
4. Convergence – converged flag set after stable_ticks_required in-band ticks at finest step
5. Reproducibility – identical seeds → identical trajectories
6. Generator rate-response – mean IEI shortens after rate increase
7. Generator reproducibility – same seed → same event sequence
"""

from __future__ import annotations

import sys
import os

# Make src/ importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import math
import random
import unittest

from adaptation.difficulty_state import DifficultyState, _log_drain
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

    def test_log_drain_key_values(self):
        # Log scale: 50 ml/min at d=-0.8, 400 ml/min at d=+1.8
        # (capped below the 600 ml/min naive-participant sustainable max)
        self.assertEqual(_log_drain(-0.8), 50)
        self.assertEqual(_log_drain(1.8), 400)
        # Monotonically increasing
        self.assertLess(_log_drain(0.0), _log_drain(0.5))
        self.assertLess(_log_drain(0.5), _log_drain(1.0))

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
            # Single-entry schedule keeps step fixed; no reversal advancement.
            step_schedule=(step,),
            cooldown_sec=cooldown_sec,
            stable_ticks_required=9999,   # disable convergence for monotone test
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
            step_schedule=(0.05,),
            stable_ticks_required=9999,
        )
        # Fill window with target score
        for t in range(1, 200):
            ctrl.push_performance(float(t), 0.70)
            delta = ctrl.tick(float(t))
            if delta is not None:
                state.update(state.d + delta)

        self.assertAlmostEqual(state.d, 0.5, places=5, msg="d changed despite on-target score")


# ---------------------------------------------------------------------------
# 3. Staircase – graduated step schedule
# ---------------------------------------------------------------------------

class TestStaircaseReversals(unittest.TestCase):

    def test_step_advances_through_schedule_on_reversals(self):
        """Step size must advance from 0.2 → 0.1 → 0.05 with each reversal."""
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=30.0,
            step_schedule=(0.20, 0.10, 0.05),
            cooldown_sec=5.0,
            stable_ticks_required=9999,   # disable convergence
        )
        state = DifficultyState(d_init=0.5)

        # Step size should start at 0.20
        self.assertAlmostEqual(ctrl.step_up, 0.20)

        t = 0.0
        dt = 1.0
        direction = "up"

        # Drive at least 2 reversals
        while ctrl.reversal_count < 2 and t < 2000:
            score = 1.0 if direction == "up" else 0.0
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
                direction = "down" if delta > 0 else "up"
            t += dt

        self.assertGreaterEqual(ctrl.reversal_count, 2)
        # After 2 reversals the schedule index should be at position 2 (step=0.05)
        self.assertAlmostEqual(ctrl.step_up, 0.05)

    def test_step_after_first_reversal(self):
        """After exactly 1 reversal step must advance to 0.10."""
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=30.0,
            step_schedule=(0.20, 0.10, 0.05),
            cooldown_sec=5.0,
            stable_ticks_required=9999,
        )
        state = DifficultyState(d_init=0.5)

        t = 0.0
        dt = 1.0
        direction = "up"

        while ctrl.reversal_count < 1 and t < 2000:
            score = 1.0 if direction == "up" else 0.0
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
                direction = "down" if delta > 0 else "up"
            t += dt

        self.assertEqual(ctrl.reversal_count, 1)
        self.assertAlmostEqual(ctrl.step_up, 0.10)

    def test_step_does_not_advance_beyond_final(self):
        """step_up must not decrease below the last schedule entry."""
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=20.0,
            step_schedule=(0.20, 0.10, 0.05),
            cooldown_sec=5.0,
            stable_ticks_required=9999,
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
            self.assertGreaterEqual(ctrl.step_up, 0.05, "step_up fell below final schedule entry")
            self.assertGreaterEqual(ctrl.step_down, 0.05, "step_down fell below final schedule entry")


# ---------------------------------------------------------------------------
# 4. Convergence detection
# ---------------------------------------------------------------------------

class TestConvergence(unittest.TestCase):
    """Convergence must only trigger after stable_ticks_required consecutive
    no-step ticks once the staircase has already reached the finest step."""

    def _build_ctrl(self, *, stable_ticks: int = 3) -> StaircaseController:
        return StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=20.0,
            min_samples=3,
            step_schedule=(0.20, 0.10, 0.05),
            cooldown_sec=5.0,
            stable_ticks_required=stable_ticks,
        )

    def _advance_to_final_step(self, ctrl, state, dt=1.0):
        """Drive alternating scores until schedule reaches the final 0.05 entry."""
        t = 0.0
        direction = "up"
        while ctrl._schedule_idx < len(ctrl._step_schedule) - 1 and t < 3000:
            score = 1.0 if direction == "up" else 0.0
            ctrl.push_performance(t, score)
            delta = ctrl.tick(t)
            if delta is not None:
                state.update(state.d + delta)
                direction = "down" if delta > 0 else "up"
            t += dt
        return t

    def test_no_convergence_before_final_step(self):
        """Stable ticks are not counted until the finest step is reached."""
        ctrl = StaircaseController(
            target_score=0.70,
            tolerance=0.05,
            window_sec=20.0,
            min_samples=3,
            step_schedule=(0.20, 0.10, 0.05),
            cooldown_sec=5.0,
            stable_ticks_required=3,
        )
        state = DifficultyState(d_init=0.5)
        # Feed on-target score before any reversal (still on first schedule entry)
        for t in range(1, 400):
            ctrl.push_performance(float(t), 0.70)
            ctrl.tick(float(t))
            if ctrl._schedule_idx > 0:
                break
        # Confirm stable counter hasn't accumulated toward convergence
        self.assertFalse(ctrl.converged)
        self.assertEqual(ctrl._stable_ticks_at_final_step, 0)

    def test_convergence_triggers_after_stable_ticks_at_final_step(self):
        """After reaching 0.05 step, 3 in-band ticks must set converged=True."""
        ctrl = self._build_ctrl(stable_ticks=3)
        state = DifficultyState(d_init=0.5)

        t = self._advance_to_final_step(ctrl, state)
        self.assertEqual(ctrl._schedule_idx, 2, "Did not reach final step")
        self.assertAlmostEqual(ctrl.step_up, 0.05)

        # Advance time by window_sec so all extreme-score samples from _advance_to_final_step
        # fall outside the rolling window before we start counting stable ticks.
        t += ctrl.window_sec + ctrl.cooldown_sec + 1
        for i in range(3):
            # Fill window with on-target scores
            for _ in range(5):
                ctrl.push_performance(t, 0.70)
                t += 1.0
            ctrl.tick(t)
            t += ctrl.cooldown_sec + 1

        self.assertTrue(ctrl.converged, "Controller did not converge after 3 stable ticks")

    def test_stability_counter_resets_on_step_at_final_step(self):
        """A step at the finest level must reset the stability counter."""
        ctrl = self._build_ctrl(stable_ticks=3)
        state = DifficultyState(d_init=0.5)
        t = self._advance_to_final_step(ctrl, state)

        # Flush stale extreme-score samples from the window before counting stable ticks.
        t += ctrl.window_sec + ctrl.cooldown_sec + 1
        for _ in range(2):
            for _ in range(5):
                ctrl.push_performance(t, 0.70)
                t += 1.0
            ctrl.tick(t)
            t += ctrl.cooldown_sec + 1
        self.assertEqual(ctrl._stable_ticks_at_final_step, 2)
        self.assertFalse(ctrl.converged)

        # Now fire a step by pushing above-target score
        t += 1.0
        for _ in range(5):
            ctrl.push_performance(t, 1.0)
            t += 1.0
        ctrl.tick(t)
        self.assertEqual(ctrl._stable_ticks_at_final_step, 0, "Counter not reset after step")

    def test_tick_always_returns_none_after_convergence(self):
        """Once converged, tick() must always return None regardless of score."""
        ctrl = self._build_ctrl(stable_ticks=3)
        state = DifficultyState(d_init=0.5)
        t = self._advance_to_final_step(ctrl, state)

        # Force convergence
        ctrl._converged = True
        for score in [0.0, 0.5, 1.0]:
            for _ in range(5):
                ctrl.push_performance(t, score)
                t += 1.0
            result = ctrl.tick(t)
            self.assertIsNone(result, f"tick() returned {result} after convergence")


# ---------------------------------------------------------------------------
# 5. Reproducibility (formerly §4)
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
            step_schedule=(0.05,),    # fixed step; reproducibility test only
            stable_ticks_required=9999,
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
# 6. PoissonEventGenerator – rate-response
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
# 7. Generator reproducibility
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
