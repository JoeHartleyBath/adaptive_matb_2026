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

    def test_alpha_high_on_confident_input(self):
        """When p is near 0 or 1 (high confidence), alpha should be near alpha_max."""
        s = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30)
        s.update(0.95)  # confidence = 2*|0.95-0.5| = 0.9
        assert s.current_alpha == pytest.approx(0.05 + 0.25 * 0.9)

    def test_alpha_low_on_uncertain_input(self):
        """When p is near 0.5 (low confidence), alpha should be near alpha_min."""
        s = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30)
        s.update(0.5)  # confidence = 0
        assert s.current_alpha == pytest.approx(0.05)

    def test_alpha_at_extreme_probability(self):
        """p = 1.0 gives confidence = 1.0, so alpha = alpha_max."""
        s = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30)
        s.update(1.0)
        assert s.current_alpha == pytest.approx(0.30)

    def test_alpha_symmetric_around_half(self):
        """p = 0.2 and p = 0.8 should produce the same alpha."""
        s1 = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30)
        s2 = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.30)
        s1.update(0.2)
        s2.update(0.8)
        assert s1.current_alpha == pytest.approx(s2.current_alpha)

    def test_smoothes_uncertain_inputs_more(self):
        """Uncertain inputs (p≈0.5) should be smoothed more heavily than
        confident ones, so the output should move less per step."""
        s_certain = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.50)
        s_uncertain = AdaptiveEmaSmoother(alpha_min=0.05, alpha_max=0.50)
        # Initialise both at 0.3
        s_certain.update(0.3)
        s_uncertain.update(0.3)
        # Step to a new value: one confident, one uncertain
        out_certain = s_certain.update(0.95)   # high confidence
        out_uncertain = s_uncertain.update(0.55)  # low confidence
        # Confident input should have moved output further from 0.3
        assert abs(out_certain - 0.3) > abs(out_uncertain - 0.3)

    def test_reset_clears_state(self):
        s = AdaptiveEmaSmoother()
        s.update(0.8)
        s.reset()
        # After reset, first sample re-initialises to that value
        assert s.update(0.2) == pytest.approx(0.2)
        # alpha reflects the confidence of the new input (|0.2-0.5|*2 = 0.6)
        expected_alpha = s.alpha_min + (s.alpha_max - s.alpha_min) * 0.6
        assert s.current_alpha == pytest.approx(expected_alpha)

    def test_invalid_alpha_range_raises(self):
        with pytest.raises(ValueError):
            AdaptiveEmaSmoother(alpha_min=0.5, alpha_max=0.3)

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
    def test_first_sample_returns_that_value(self):
        s = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.1)
        out = s.update(0.6)
        assert out == pytest.approx(0.6, abs=0.01)

    def test_lag_delays_step_response(self):
        """After a step from 0 to 1, the lagged output should stay near 0
        for at least lag_n steps because it estimates x_{t-lag_n}."""
        s = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.1)
        # Feed 10 zeros, then switch to 1
        for _ in range(10):
            s.update(0.0)
        # First step after switch — lagged output should still be near 0
        out = s.update(1.0)
        assert out < 0.3

    def test_converges_toward_constant(self):
        """After many constant inputs, output converges to that value."""
        s = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.1)
        for _ in range(200):
            out = s.update(0.7)
        assert out == pytest.approx(0.7, abs=0.05)

    def test_output_clamped_to_unit_interval(self):
        """Output stays in [0, 1] even with extreme inputs."""
        s = FixedLagSmoother(lag_n=2, process_noise=0.01, measurement_noise=0.05)
        for v in [2.0, -1.0, 3.0, -0.5, 1.5]:
            out = s.update(v)
            assert 0.0 <= out <= 1.0

    def test_higher_measurement_noise_smooths_more(self):
        """With higher R, output should react less to a sudden step."""
        s_low_r = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.05)
        s_high_r = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.5)
        for _ in range(10):
            s_low_r.update(0.0)
            s_high_r.update(0.0)
        # Push a step to 1
        for _ in range(3):
            out_low = s_low_r.update(1.0)
            out_high = s_high_r.update(1.0)
        # Low R should have tracked faster
        assert out_low > out_high

    def test_reset_clears_state(self):
        s = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.1)
        for _ in range(10):
            s.update(0.9)
        s.reset()
        out = s.update(0.1)
        assert out == pytest.approx(0.1, abs=0.01)

    def test_negative_lag_raises(self):
        with pytest.raises(ValueError):
            FixedLagSmoother(lag_n=-1)

    def test_zero_process_noise_raises(self):
        with pytest.raises(ValueError):
            FixedLagSmoother(process_noise=0.0)

    def test_zero_measurement_noise_raises(self):
        with pytest.raises(ValueError):
            FixedLagSmoother(measurement_noise=0.0)

    def test_zero_lag_returns_filtered_estimate(self):
        """With lag_n=0, the smoother reduces to a Kalman filter (no RTS pass)."""
        s = FixedLagSmoother(lag_n=0, process_noise=0.005, measurement_noise=0.1)
        out = s.update(0.5)
        assert out == pytest.approx(0.5, abs=0.01)
        # Should still smooth
        s.update(0.5)
        out = s.update(1.0)
        assert out < 0.9  # filtered, not raw

    def test_smoother_reduces_variance(self):
        """Smoothed output should have lower variance than raw input for
        noisy signal around a constant mean."""
        import random
        rng = random.Random(42)
        s = FixedLagSmoother(lag_n=4, process_noise=0.005, measurement_noise=0.1)
        raw_vals = [0.5 + 0.2 * (rng.random() - 0.5) for _ in range(100)]
        smoothed_vals = [s.update(v) for v in raw_vals]
        raw_var = sum((v - 0.5) ** 2 for v in raw_vals) / len(raw_vals)
        sm_var = sum((v - 0.5) ** 2 for v in smoothed_vals[20:]) / len(smoothed_vals[20:])
        assert sm_var < raw_var


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
        cfg = MwlSmootherConfig(method="fixed_lag", lag_n=4,
                                process_noise=0.01, measurement_noise=0.2)
        s = make_smoother(cfg)
        assert isinstance(s, FixedLagSmoother)
        assert s.lag_n == 4
        assert s.process_noise == 0.01
        assert s.measurement_noise == 0.2

    def test_unknown_method_raises(self):
        cfg = MwlSmootherConfig(method="unknown")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            make_smoother(cfg)

    def test_default_config_produces_ema(self):
        s = make_smoother(MwlSmootherConfig())
        assert isinstance(s, EmaSmoother)
        assert s.alpha == pytest.approx(0.10)
