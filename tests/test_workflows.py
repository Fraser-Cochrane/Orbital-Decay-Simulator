"""Tests for the isolated approximate demonstration profile."""

import unittest

import numpy as np

from velox_decay import model
from velox_decay.config import SimulationRequest, SpacecraftParameters
from velox_decay.plotting import EXPORT_PNG_DPI, QUICK_DEMO_PNG_DPI
from velox_decay.workflows import (
    QUICK_DEMO_MONTE_CARLO_RUNS,
    QUICK_DEMO_PROPAGATION_STEP,
    _completion_message,
    _simulation_profile,
    _spacecraft_profile,
    heatmap_phase_epochs,
)


class QuickDemoTests(unittest.TestCase):
    """Verify quick-demo isolation and workflow utility behavior."""

    def test_demo_png_resolution_is_high(self) -> None:
        """All exported graphs use the documented high-resolution setting."""
        self.assertEqual(EXPORT_PNG_DPI, 600)
        self.assertEqual(QUICK_DEMO_PNG_DPI, EXPORT_PNG_DPI)

    def test_profile_is_temporary(self) -> None:
        """Demo numerical settings are restored when their context exits."""
        full_runs = model.MONTE_CARLO_RUNS
        full_step = model.PROPAGATION_STEP

        with _simulation_profile(True):
            self.assertEqual(
                model.MONTE_CARLO_RUNS, QUICK_DEMO_MONTE_CARLO_RUNS
            )
            self.assertEqual(
                model.PROPAGATION_STEP, QUICK_DEMO_PROPAGATION_STEP
            )

        self.assertEqual(model.MONTE_CARLO_RUNS, full_runs)
        self.assertEqual(model.PROPAGATION_STEP, full_step)

    def test_cached_demo_message_is_explicit(self) -> None:
        """Completion text clearly identifies an exact demo-cache hit."""
        request = SimulationRequest.from_text(
            "2024-01-01", "500,600", "line", True, True
        )
        message = _completion_message(request, "Graph generated.", 1, 1)
        self.assertIn("loaded entirely from exact cache", message)

    def test_demo_and_full_cache_keys_are_distinct(self) -> None:
        """Coarse and scientific runs cannot share cached numerical results."""
        epoch = np.datetime64("2024-01-01T00:00:00", "s")
        with _simulation_profile(False):
            full_key = model.result_cache_key(epoch, model.SIMULATION_DURATION)
        with _simulation_profile(True):
            demo_key = model.result_cache_key(epoch, model.SIMULATION_DURATION)
        self.assertNotEqual(full_key, demo_key)

    def test_default_heatmap_date_reproduces_reference_epochs(self) -> None:
        """The default heat-map date maps to the configured phase epochs."""
        actual = heatmap_phase_epochs(model.SOLAR_CYCLE_MINIMUM_EPOCH)
        np.testing.assert_array_equal(actual, model.solar_cycle_phase_epochs())

    def test_spacecraft_profile_is_temporary_and_changes_cache_key(self) -> None:
        """Custom spacecraft settings are isolated and included in cache keys."""
        epoch = np.datetime64("2024-01-01T00:00:00", "s")
        original_mass = model.SATELLITE_MASS
        default_key = model.result_cache_key(epoch, model.SIMULATION_DURATION)
        custom = SpacecraftParameters(mass_kg=100.0)

        with _spacecraft_profile(custom):
            self.assertEqual(model.SATELLITE_MASS, 100.0)
            custom_key = model.result_cache_key(epoch, model.SIMULATION_DURATION)

        self.assertEqual(model.SATELLITE_MASS, original_mass)
        self.assertNotEqual(default_key, custom_key)

    def test_smooth_trend_preserves_a_decreasing_sample_shape(self) -> None:
        """The display curve remains smooth and monotonic between samples."""
        heights = np.array([500.0, 550.0, 600.0, 650.0, 700.0])
        rates = np.array([80.0, 40.0, 20.0, 10.0, 5.0])
        smooth_heights, smooth_rates = model.smooth_trend_curve(
            heights,
            rates,
        )

        self.assertEqual(smooth_heights.size, 401)
        self.assertTrue(np.all(np.diff(smooth_heights) > 0.0))
        self.assertTrue(np.all(np.diff(smooth_rates) <= 0.0))
        np.testing.assert_allclose(smooth_rates[::100], rates)

    def test_exponential_regression_is_not_forced_through_samples(self) -> None:
        """The fitted altitude trend minimizes residuals without interpolation."""
        heights = np.array([500.0, 550.0, 600.0, 650.0, 700.0])
        rates = np.array([82.0, 37.0, 21.0, 9.0, 5.5])
        sigma = np.array([12.0, 5.0, 3.0, 1.5, 0.8])
        regression_heights, regression_rates = (
            model.exponential_regression_curve(heights, rates, sigma)
        )

        self.assertEqual(regression_heights.size, 401)
        self.assertTrue(np.all(np.diff(regression_rates) < 0.0))
        self.assertFalse(np.allclose(regression_rates[::100], rates))
        np.testing.assert_allclose(
            np.diff(np.log(regression_rates), n=2),
            0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
