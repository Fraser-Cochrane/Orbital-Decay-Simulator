"""Tests for browser payload validation and HTTP job coordination."""

from pathlib import Path
from unittest import mock
import unittest

from fastapi import HTTPException, Response

from velox_decay import web


class WebInterfaceTests(unittest.TestCase):
    """Verify the web adapter without executing a numerical propagation."""

    def tearDown(self) -> None:
        """Remove job records created by an individual unit test."""
        with web._jobs_lock:
            web._jobs.clear()

    def test_frontend_assets_are_available(self) -> None:
        """The HTTP adapter resolves every required browser asset."""
        self.assertTrue((web.WEB_ROOT / "index.html").is_file())
        self.assertTrue((web.WEB_ROOT / "styles.css").is_file())
        self.assertTrue((web.WEB_ROOT / "app.js").is_file())
        page = web.website()
        self.assertEqual(Path(page.path), web.WEB_ROOT / "index.html")
        self.assertEqual(page.headers["cache-control"], "no-cache")

    def test_frontend_includes_connection_recovery(self) -> None:
        """The static client offers recovery after its Python service restarts."""
        page = (web.WEB_ROOT / "index.html").read_text(encoding="utf-8")
        script = (web.WEB_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="reconnect-button"', page)
        self.assertIn(
            'elements.reconnectButton.addEventListener("click", initializeApplication)',
            script,
        )

    def test_frontend_api_urls_support_a_subdirectory_mount(self) -> None:
        """Browser API requests remain relative to the application's mount path."""
        script = (web.WEB_ROOT / "app.js").read_text(encoding="utf-8")
        self.assertIn('fetch("api/config"', script)
        self.assertIn('fetch("api/jobs"', script)
        self.assertNotIn('fetch("/api/', script)

    def test_configuration_exposes_supported_dates_and_defaults(self) -> None:
        """The browser receives model-derived date limits and spacecraft data."""
        response = Response()
        payload = web.configuration(response)
        self.assertIn("line", payload["latest_launch_dates"])
        self.assertIn("heatmap", payload["latest_launch_dates"])
        self.assertEqual(payload["altitude_sample_count"], 20)
        self.assertIs(payload["allow_full_runs"], web.ALLOW_FULL_RUNS)
        self.assertEqual(payload["spacecraft"]["mass_kg"], 123.0)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_valid_job_is_queued_and_reported(self) -> None:
        """A valid request enters the serialized queue and exposes live status."""
        payload = web.SimulationInput(
            launch_date="2024-01-01",
            lower_altitude_km=500.0,
            upper_altitude_km=700.0,
        )
        response = Response()
        with mock.patch.object(web._executor, "submit") as submit:
            record = web.create_job(payload, response)

        submit.assert_called_once()
        self.assertEqual(record["state"], "queued")
        self.assertEqual(response.status_code, 202)
        status_response = Response()
        current = web.job_status(record["job_id"], status_response)
        self.assertEqual(current["progress"], 0.0)
        self.assertEqual(status_response.headers["cache-control"], "no-store")

    def test_reversed_altitudes_are_rejected(self) -> None:
        """Canonical simulation validation rejects a reversed altitude range."""
        payload = web.SimulationInput(
            launch_date="2024-01-01",
            lower_altitude_km=650.0,
            upper_altitude_km=550.0,
        )
        with self.assertRaises(HTTPException) as context:
            web.create_job(payload, Response())
        self.assertEqual(context.exception.status_code, 422)

    def test_public_profile_rejects_a_full_scientific_run(self) -> None:
        """The SRCF profile can keep computationally expensive runs local."""
        payload = web.SimulationInput(
            launch_date="2024-01-01",
            lower_altitude_km=500.0,
            upper_altitude_km=700.0,
            quick_demo=False,
        )
        with mock.patch.object(web, "ALLOW_FULL_RUNS", False):
            with self.assertRaises(HTTPException) as context:
                web.create_job(payload, Response())
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
