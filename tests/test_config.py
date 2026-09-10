"""Fast validation tests that do not run the full physical propagation."""

import unittest

import numpy as np

from velox_decay.config import (
    ALTITUDE_SAMPLE_COUNT,
    SimulationRequest,
    SpacecraftParameters,
    VELOX_C1_DEFAULTS,
    latest_supported_launch_date,
    parse_altitude_bounds,
    parse_altitudes,
)


class ConfigurationTests(unittest.TestCase):
    """Verify public configuration parsing and validation behavior."""

    def test_default_altitude_range_is_inclusive(self) -> None:
        """Range syntax includes both requested altitude endpoints."""
        values = parse_altitudes("500:700:10")
        np.testing.assert_array_equal(values, np.arange(500.0, 701.0, 10.0))

    def test_altitude_list_is_sorted_and_unique(self) -> None:
        """Explicit altitude lists are normalized into sorted unique values."""
        self.assertEqual(parse_altitudes("600, 500, 600"), (500.0, 600.0))

    def test_altitude_bounds_generate_twenty_inclusive_values(self) -> None:
        """GUI altitude bounds produce the configured number of samples."""
        values = parse_altitude_bounds("500", "700")
        self.assertEqual(len(values), ALTITUDE_SAMPLE_COUNT)
        self.assertEqual(values[0], 500.0)
        self.assertEqual(values[-1], 700.0)

    def test_reversed_altitude_bounds_are_rejected(self) -> None:
        """A lower bound above the upper bound is invalid."""
        with self.assertRaisesRegex(ValueError, "Upper altitude"):
            parse_altitude_bounds("600", "550")

    def test_request_normalizes_launch_date(self) -> None:
        """Line-graph dates are normalized to second precision."""
        request = SimulationRequest.from_text("2024-01-01", "500,600", "line")
        self.assertEqual(request.launch_date, "2024-01-01T00:00:00")

    def test_heatmap_normalizes_selected_start_date(self) -> None:
        """Heat-map dates are normalized to second precision."""
        request = SimulationRequest.from_text(
            "2020-02-03", "500,600", "heatmap"
        )
        self.assertEqual(request.launch_date, "2020-02-03T00:00:00")

    def test_line_date_beyond_space_weather_coverage_is_rejected(self) -> None:
        """A line run cannot exceed the local geomagnetic-data interval."""
        too_late = latest_supported_launch_date("line") + np.timedelta64(1, "D")
        with self.assertRaisesRegex(ValueError, "too late"):
            SimulationRequest.from_text(str(too_late), "500,600", "line")

    def test_heatmap_date_beyond_space_weather_coverage_is_rejected(self) -> None:
        """A heat map cannot exceed the local geomagnetic-data interval."""
        too_late = latest_supported_launch_date("heatmap") + np.timedelta64(1, "D")
        with self.assertRaisesRegex(ValueError, "too late"):
            SimulationRequest.from_text(str(too_late), "500,600", "heatmap")

    def test_request_from_bounds_uses_configured_sample_count(self) -> None:
        """Bound-based requests contain the configured altitude count."""
        request = SimulationRequest.from_bounds(
            "2024-01-01", "500", "700", "line"
        )
        self.assertEqual(len(request.altitudes_km), ALTITUDE_SAMPLE_COUNT)

    def test_quick_demo_is_explicitly_opt_in(self) -> None:
        """Scientific settings remain active unless demo mode is selected."""
        full_request = SimulationRequest.from_text(
            "2024-01-01", "500,600", "line"
        )
        demo_request = SimulationRequest.from_text(
            "2024-01-01", "500,600", "line", True, True
        )
        self.assertFalse(full_request.quick_demo)
        self.assertTrue(demo_request.quick_demo)

    def test_velox_c1_spacecraft_defaults_are_used(self) -> None:
        """Requests use VELOX-C1 values when no custom spacecraft is supplied."""
        request = SimulationRequest.from_text(
            "2024-01-01", "500,600", "line"
        )
        self.assertEqual(request.spacecraft, VELOX_C1_DEFAULTS)

    def test_custom_spacecraft_values_are_validated(self) -> None:
        """Custom values are parsed and angular values are normalized."""
        parameters = SpacecraftParameters.from_text(
            "100", "0.4", "0.5", "0.6", "0.7", "0.45", "0.4",
            "0.01", "51.6", "370", "-10"
        )
        self.assertEqual(parameters.mass_kg, 100.0)
        self.assertEqual(parameters.raan_deg, 10.0)
        self.assertEqual(parameters.argument_of_perigee_deg, 350.0)

    def test_nonpositive_mass_is_rejected(self) -> None:
        """Spacecraft mass must be physically positive."""
        with self.assertRaisesRegex(ValueError, "Mass must be greater than zero"):
            SpacecraftParameters.from_text(
                "0", "0.4", "0.5", "0.6", "0.7", "0.45", "0.4",
                "0.01", "51.6", "0", "0"
            )


if __name__ == "__main__":
    unittest.main()
