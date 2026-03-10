"""Tests for src/adaptation/mwl_smoother.py

Run with:
    pytest tests/test_mwl_smoother.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptation.mwl_smoother import (
    AdaptiveEmaSmoother,
    EmaSmoother,
    FixedLagSmoother,
    MwlSmootherConfig,
    SmaSmoother,
    make_smoother,
)


# ---------------------------------------------------------------------------
# EmaSmoother
# ---------------------------------------------------------------------------

class TestEmaSmoother:
    def test_first_sample_initialises_state(self):
        s = EmaSmoother(alpha=0.5)
        assert s.update(0.8) == pytest.approx(0.8)

    def test_converges_toward_constant_input(self):
        s = EmaSmoother(alpha=0.1)
        for _ in range(200):
            out = s.update(1.0)
        assert out == pytest.approx(1.0, abs=1e-3)

    def test_step_response_direction(self):
        """Output increases monotonically when input jumps from 0 to 1."""
        s = EmaSmoother(alpha=0.2)
        s.update(0.0)
        prev = 0.0
        for _ in range(20):
            out = s.update(1.0)
            assert out >= prev
            prev = out

    def test_alpha_1_is_identity(self):
        s = EmaSmoother(alpha=1.0)
        for v in [0.1, 0.9, 0.3, 0.7]:
            assert s.update(v) == pytest.approx(v)

    def test_reset_clears_state(self):
        s = EmaSmoother(alpha=0.5)
        s.update(1.0)
        s.reset()
        # After reset, next sample should re-initialise to that value
        assert s.update(0.3) == pytest.approx(0.3)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            EmaSmoother(alpha=0.0)
        with pytest.raises(ValueError):
            EmaSmoother(alpha=1.1)

    def test_ema_formula(self):
        """Verify the numeric formula directly for two steps."""
        alpha = 0.3
        s = EmaSmoother(alpha=alpha)
        x0, x1 = 0.6, 0.2
        s0 = s.update(x0)
        assert s0 == pytest.approx(x0)
        s1 = s.update(x1)
        assert s1 == pytest.approx(alpha * x1 + (1 - alpha) * x0)


# ---------------------------------------------------------------------------
# SmaSmoother
# ---------------------------------------------------------------------------

class TestSmaSmoother:
    def test_single_sample(self):
        s = SmaSmoother(window_n=4)
        assert s.update(0.5) == pytest.approx(0.5)

    def test_mean_of_window(self):
        s = SmaSmoother(window_n=4)
        for v in [0.2, 0.4, 0.6, 0.8]:
            out = s.update(v)
        assert out == pytest.approx(0.5)

    def test_oldest_sample_evicted(self):
        s = SmaSmoother(window_n=3)
        s.update(1.0)
        s.update(1.0)
        s.update(1.0)
        # Window is [1, 1, 1]; push 0 → window becomes [1, 1, 0]
        out = s.update(0.0)
        assert out == pytest.approx(2.0 / 3.0)

    def test_reset_empties_buffer(self):
        s = SmaSmoother(window_n=4)
        s.update(0.9)
        s.reset()
        assert s.update(0.1) == pytest.approx(0.1)

    def test_window_n_1_is_identity(self):
        s = SmaSmoother(window_n=1)
        for v in [0.2, 0.5, 0.8]:
            assert s.update(v) == pytest.approx(v)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SmaSmoother(window_n=0)


# ---------------------------------------------------------------------------
# AdaptiveEmaSmoother
# ---------------------------------------------------------------------------

class TestAdaptiveEmaSmoother:
    def test_first_sample_initialises_state(self):
        s = AdaptiveEmaSmoother()
        assert s.update(0.7) == pytest.approx(0.7)

    def test_alpha_low_on_stable_signal(self):
        """After many identical inputs, current_alpha should be near alpha_min."""
        s = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30, var_ceiling=0.04)
        for _ in range(30):
            s.update(0.5)
        assert s.current_alpha == pytest.approx(s.alpha_min, abs=1e-6)

    def test_alpha_high_on_volatile_signal(self):
        """Alternating 0/1 signal should push alpha toward alpha_max."""
        s = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30, var_ceiling=0.04)
        for i in range(30):
            s.update(float(i % 2))
        assert s.current_alpha > s.alpha_min

    def test_reset_clears_state(self):
        s = AdaptiveEmaSmoother()
        s.update(0.8)
        s.reset()
        assert s.update(0.2) == pytest.approx(0.2)
        assert s.current_alpha == pytest.approx(s.alpha_min)

    def test_invalid_alpha_range_raises(self):
        with pytest.raises(ValueError):
            AdaptiveEmaSmoother(alpha_min=0.5, alpha_max=0.3)

    def test_invalid_variance_window_raises(self):
        with pytest.raises(ValueError):
            AdaptiveEmaSmoother(variance_window_n=1)

    def test_output_stays_bounded(self):
        """Smoothed output must remain in [0, 1] when inputs are in [0, 1]."""
        s = AdaptiveEmaSmoother()
        import random
        rng = random.Random(42)
        for _ in range(100):
            out = s.update(rng.random())
            assert 0.0 <= out <= 1.0


# ---------------------------------------------------------------------------
# FixedLagSmoother
# ---------------------------------------------------------------------------

class TestFixedLagSmoother:
    def test_output_before_buffer_fills(self):
        """Before lag depth is reached, returns mean of all available samples."""
        s = FixedLagSmoother(window_n=9, lag_n=4)
        # Only 2 samples — centre would be at index -3 (negative), so mean all
        out = s.update(0.4)
        assert out == pytest.approx(0.4)
        out = s.update(0.6)
        assert out == pytest.approx(0.5)

    def test_lag_provides_delayed_smoothing(self):
        """The lag means output stays near 0 for the first few HIGH inputs."""
        s = FixedLagSmoother(window_n=9, lag_n=4)
        # Feed a step: 0 for first 5 samples, then 1 forever.
        # On the 6th sample (index 5, first 1 pushed):
        #   buf = [0,0,0,0,0,1], n=6, centre = 6-1-4 = 1
        #   window = buf[0:6] = [0,0,0,0,0,1]  → mean = 1/6 ≈ 0.167
        # So output should be well below 0.5 immediately after the step.
        outs = []
        for _ in range(5):
            outs.append(s.update(0.0))
        for _ in range(10):
            outs.append(s.update(1.0))
        # First sample after step — lag centre still deep in the 0 region
        assert outs[5] < 0.5
        # Many samples later — output should have caught up to ~1.0
        assert outs[-1] > 0.9

    def test_reset_clears_buffer(self):
        s = FixedLagSmoother(window_n=9, lag_n=4)
        for _ in range(9):
            s.update(0.9)
        s.reset()
        out = s.update(0.1)
        assert out == pytest.approx(0.1)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            FixedLagSmoother(window_n=8, lag_n=4)  # needs window_n > 2*lag_n

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            FixedLagSmoother(window_n=9, lag_n=-1)

    def test_zero_lag_is_identity(self):
        """lag_n=0 centres the window on the newest sample with ±0 radius.

        window_slice = buf[centre:centre+1] = [newest] → returns newest.
        This is the degenerate case: no lag, no smoothing.
        """
        s = FixedLagSmoother(window_n=3, lag_n=0)
        for v in [0.2, 0.5, 0.8, 0.3, 0.6]:
            assert s.update(v) == pytest.approx(v)


# ---------------------------------------------------------------------------
# make_smoother factory
# ---------------------------------------------------------------------------

class TestMakeSmooother:
    def test_ema(self):
        cfg = MwlSmootherConfig(method="ema", alpha=0.2)
        s = make_smoother(cfg)
        assert isinstance(s, EmaSmoother)
        assert s.alpha == 0.2

    def test_sma(self):
        cfg = MwlSmootherConfig(method="sma", window_n=12)
        s = make_smoother(cfg)
        assert isinstance(s, SmaSmoother)
        assert s.window_n == 12

    def test_adaptive_ema(self):
        cfg = MwlSmootherConfig(method="adaptive_ema", alpha_min=0.05, alpha_max=0.25)
        s = make_smoother(cfg)
        assert isinstance(s, AdaptiveEmaSmoother)
        assert s.alpha_min == 0.05
        assert s.alpha_max == 0.25

    def test_fixed_lag(self):
        cfg = MwlSmootherConfig(method="fixed_lag", window_n=9, lag_n=4)
        s = make_smoother(cfg)
        assert isinstance(s, FixedLagSmoother)
        assert s.lag_n == 4

    def test_unknown_method_raises(self):
        cfg = MwlSmootherConfig(method="unknown")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            make_smoother(cfg)

    def test_default_config_produces_ema(self):
        s = make_smoother(MwlSmootherConfig())
        assert isinstance(s, EmaSmoother)
        assert s.alpha == pytest.approx(0.10)
