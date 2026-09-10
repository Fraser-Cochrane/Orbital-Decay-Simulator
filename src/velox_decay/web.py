"""HTTP interface for running orbital decay workflows from a web browser."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sysconfig
import threading
from typing import Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .config import (
    SimulationRequest,
    SpacecraftParameters,
    VELOX_C1_DEFAULTS,
    latest_supported_launch_date,
)
from .workflows import RunResult, run_request


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_web_root() -> Path:
    """Locate browser assets in a source checkout or deployed application."""
    configured = os.environ.get("VELOX_WEB_ROOT")
    candidates = (
        [Path(configured)]
        if configured
        else [
            PROJECT_ROOT / "docs",
            Path(sysconfig.get_path("data")) / "share" / "velox_decay_web",
            Path.cwd() / "docs",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "index.html").is_file():
            return resolved
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise RuntimeError(
        f"Web assets were not found in: {searched}. Set VELOX_WEB_ROOT to "
        "the directory containing index.html."
    )


WEB_ROOT = _resolve_web_root()
OUTPUT_ROOT = Path(
    os.environ.get("VELOX_DECAY_OUTPUT_DIR", PROJECT_ROOT / "outputs")
).resolve()
WEB_OUTPUT_ROOT = OUTPUT_ROOT / "web"
MAX_ACTIVE_JOBS = int(os.environ.get("VELOX_WEB_MAX_ACTIVE_JOBS", "4"))
MAX_JOB_RECORDS = int(os.environ.get("VELOX_WEB_MAX_JOB_RECORDS", "100"))
ALLOW_FULL_RUNS = os.environ.get("VELOX_WEB_ALLOW_FULL_RUNS", "1").casefold() in {
    "1",
    "true",
    "yes",
    "on",
}

JobState = Literal["queued", "running", "complete", "failed"]


class SpacecraftInput(BaseModel):
    """Validated browser fields for spacecraft and starting-orbit values."""

    model_config = ConfigDict(extra="forbid")

    mass_kg: float = Field(default=VELOX_C1_DEFAULTS.mass_kg, gt=0.0)
    aerodynamic_area_m2: float = Field(
        default=VELOX_C1_DEFAULTS.aerodynamic_area_m2,
        gt=0.0,
    )
    body_x_m: float = Field(default=VELOX_C1_DEFAULTS.body_dimensions_m[0], gt=0.0)
    body_y_m: float = Field(default=VELOX_C1_DEFAULTS.body_dimensions_m[1], gt=0.0)
    body_z_m: float = Field(default=VELOX_C1_DEFAULTS.body_dimensions_m[2], gt=0.0)
    solar_radiation_area_m2: float = Field(
        default=VELOX_C1_DEFAULTS.solar_radiation_area_m2,
        gt=0.0,
    )
    solar_reflection_factor: float = Field(
        default=VELOX_C1_DEFAULTS.solar_reflection_factor,
        ge=0.0,
        le=1.0,
    )
    eccentricity: float = Field(
        default=VELOX_C1_DEFAULTS.eccentricity,
        ge=0.0,
        lt=1.0,
    )
    inclination_deg: float = Field(
        default=VELOX_C1_DEFAULTS.inclination_deg,
        ge=0.0,
        le=180.0,
    )
    raan_deg: float = VELOX_C1_DEFAULTS.raan_deg
    argument_of_perigee_deg: float = VELOX_C1_DEFAULTS.argument_of_perigee_deg

    def to_parameters(self) -> SpacecraftParameters:
        """Convert browser values through the package's canonical validator."""
        return SpacecraftParameters.from_text(
            mass_kg=str(self.mass_kg),
            aerodynamic_area_m2=str(self.aerodynamic_area_m2),
            body_x_m=str(self.body_x_m),
            body_y_m=str(self.body_y_m),
            body_z_m=str(self.body_z_m),
            solar_radiation_area_m2=str(self.solar_radiation_area_m2),
            solar_reflection_factor=str(self.solar_reflection_factor),
            eccentricity=str(self.eccentricity),
            inclination_deg=str(self.inclination_deg),
            raan_deg=str(self.raan_deg),
            argument_of_perigee_deg=str(self.argument_of_perigee_deg),
        )


class SimulationInput(BaseModel):
    """Validated JSON payload accepted by the graph-generation endpoint."""

    model_config = ConfigDict(extra="forbid")

    launch_date: str
    lower_altitude_km: float = Field(ge=500.0, le=700.0)
    upper_altitude_km: float = Field(ge=500.0, le=700.0)
    plot_kind: Literal["line", "heatmap"] = "line"
    use_cache: bool = True
    quick_demo: bool = True
    spacecraft: SpacecraftInput = Field(default_factory=SpacecraftInput)

    def to_request(self) -> SimulationRequest:
        """Create the same validated request used by the desktop interface."""
        return SimulationRequest.from_bounds(
            launch_date=self.launch_date,
            lower_altitude=str(self.lower_altitude_km),
            upper_altitude=str(self.upper_altitude_km),
            plot_kind=self.plot_kind,
            use_cache=self.use_cache,
            quick_demo=self.quick_demo,
            spacecraft=self.spacecraft.to_parameters(),
        )


@dataclass(slots=True)
class WebJob:
    """Mutable server-side state for one queued simulation request."""

    identifier: str
    request: SimulationRequest
    state: JobState = "queued"
    progress: float = 0.0
    message: str = "Waiting for the simulation worker"
    output_path: Path | None = None
    error: str | None = None
    created_at: str = ""

    def public_record(self) -> dict[str, object]:
        """Return the browser-safe status representation of this job."""
        record: dict[str, object] = {
            "job_id": self.identifier,
            "state": self.state,
            "progress": round(self.progress, 1),
            "message": self.message,
            "created_at": self.created_at,
        }
        if self.error is not None:
            record["error"] = self.error
        if self.state == "complete":
            record["image_url"] = f"api/jobs/{self.identifier}/image"
        return record


app = FastAPI(
    title="VELOX-C1 Orbital Decay Web API",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")

_jobs: dict[str, WebJob] = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="velox-web")


def _utc_timestamp() -> str:
    """Return a compact UTC timestamp for browser job metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_record(identifier: str) -> WebJob:
    """Resolve a job identifier or raise an HTTP not-found response."""
    with _jobs_lock:
        job = _jobs.get(identifier)
        if job is None:
            raise HTTPException(status_code=404, detail="Simulation job not found.")
        return job


def _active_job_count() -> int:
    """Count queued and running jobs while holding the registry lock."""
    return sum(job.state in ("queued", "running") for job in _jobs.values())


def _prune_job_records() -> None:
    """Bound memory use by discarding the oldest inactive job records."""
    inactive = [
        job
        for job in _jobs.values()
        if job.state in ("complete", "failed")
    ]
    inactive.sort(key=lambda job: job.created_at)
    excess = max(0, len(_jobs) - MAX_JOB_RECORDS + 1)
    for job in inactive[:excess]:
        _jobs.pop(job.identifier, None)


def _update_progress(identifier: str, percentage: float, message: str) -> None:
    """Update a running job from the numerical workflow callback."""
    with _jobs_lock:
        job = _jobs.get(identifier)
        if job is None:
            return
        job.state = "running"
        job.progress = max(0.0, min(float(percentage), 100.0))
        job.message = message


def _execute_job(identifier: str) -> None:
    """Run one job in the serialized numerical worker and store its result."""
    with _jobs_lock:
        job = _jobs.get(identifier)
        if job is None:
            return
        request = job.request
        job.state = "running"
        job.message = "Starting simulation"

    try:
        result: RunResult = run_request(
            request,
            WEB_OUTPUT_ROOT / identifier,
            progress=lambda percentage, message: _update_progress(
                identifier,
                percentage,
                message,
            ),
        )
    except Exception as error:
        with _jobs_lock:
            job = _jobs.get(identifier)
            if job is not None:
                job.state = "failed"
                job.error = str(error) or error.__class__.__name__
                job.message = "The graph could not be generated"
        return

    with _jobs_lock:
        job = _jobs.get(identifier)
        if job is not None:
            job.state = "complete"
            job.progress = 100.0
            job.message = result.message
            job.output_path = result.output_path.resolve()


def _spacecraft_defaults() -> dict[str, float]:
    """Return VELOX-C1 values using names shared with the browser form."""
    return {
        "mass_kg": VELOX_C1_DEFAULTS.mass_kg,
        "aerodynamic_area_m2": VELOX_C1_DEFAULTS.aerodynamic_area_m2,
        "body_x_m": VELOX_C1_DEFAULTS.body_dimensions_m[0],
        "body_y_m": VELOX_C1_DEFAULTS.body_dimensions_m[1],
        "body_z_m": VELOX_C1_DEFAULTS.body_dimensions_m[2],
        "solar_radiation_area_m2": VELOX_C1_DEFAULTS.solar_radiation_area_m2,
        "solar_reflection_factor": VELOX_C1_DEFAULTS.solar_reflection_factor,
        "eccentricity": VELOX_C1_DEFAULTS.eccentricity,
        "inclination_deg": VELOX_C1_DEFAULTS.inclination_deg,
        "raan_deg": VELOX_C1_DEFAULTS.raan_deg,
        "argument_of_perigee_deg": VELOX_C1_DEFAULTS.argument_of_perigee_deg,
    }


@app.get("/", include_in_schema=False)
def website() -> FileResponse:
    """Serve the interactive project page from the application origin."""
    return FileResponse(
        WEB_ROOT / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/styles.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    """Serve the project's visual design rules."""
    return FileResponse(
        WEB_ROOT / "styles.css",
        media_type="text/css",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/app.js", include_in_schema=False)
def browser_application() -> FileResponse:
    """Serve the dependency-free browser interaction code."""
    return FileResponse(
        WEB_ROOT / "app.js",
        media_type="text/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    """Expose a lightweight health probe for deployment platforms."""
    return {"status": "ok"}


@app.get("/api/config")
def configuration(response: Response) -> dict[str, object]:
    """Return date limits, model bounds, and VELOX-C1 form defaults."""
    response.headers["Cache-Control"] = "no-store"
    line_latest = np.datetime_as_string(
        latest_supported_launch_date("line"),
        unit="D",
    )
    heatmap_latest = np.datetime_as_string(
        latest_supported_launch_date("heatmap"),
        unit="D",
    )
    return {
        "minimum_launch_date": "1961-01-01",
        "latest_launch_dates": {
            "line": line_latest,
            "heatmap": heatmap_latest,
        },
        "default_launch_dates": {
            "line": min("2024-01-01", line_latest),
            "heatmap": min("2019-12-01", heatmap_latest),
        },
        "altitude_bounds_km": {"minimum": 500.0, "maximum": 700.0},
        "altitude_sample_count": 20,
        "allow_full_runs": ALLOW_FULL_RUNS,
        "spacecraft": _spacecraft_defaults(),
    }


@app.post("/api/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_job(payload: SimulationInput, response: Response) -> dict[str, object]:
    """Validate, enqueue, and return one browser simulation job."""
    if not payload.quick_demo and not ALLOW_FULL_RUNS:
        raise HTTPException(
            status_code=403,
            detail=(
                "Full scientific runs are disabled on this public service. "
                "Use quick demo or run the package locally."
            ),
        )
    try:
        request = payload.to_request()
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    identifier = uuid4().hex
    with _jobs_lock:
        if _active_job_count() >= MAX_ACTIVE_JOBS:
            raise HTTPException(
                status_code=429,
                detail="The simulation queue is full. Try again after a job finishes.",
            )
        _prune_job_records()
        job = WebJob(
            identifier=identifier,
            request=request,
            created_at=_utc_timestamp(),
        )
        _jobs[identifier] = job
        record = job.public_record()

    _executor.submit(_execute_job, identifier)
    response.status_code = status.HTTP_202_ACCEPTED
    response.headers["Cache-Control"] = "no-store"
    response.headers["Location"] = f"api/jobs/{identifier}"
    return record


@app.get("/api/jobs/{identifier}")
def job_status(identifier: str, response: Response) -> dict[str, object]:
    """Return live progress and completion metadata for a simulation job."""
    job = _job_record(identifier)
    with _jobs_lock:
        record = job.public_record()
    response.headers["Cache-Control"] = "no-store"
    return record


@app.get("/api/jobs/{identifier}/image")
def job_image(identifier: str) -> FileResponse:
    """Return a completed PNG while preventing arbitrary file access."""
    job = _job_record(identifier)
    with _jobs_lock:
        if job.state != "complete" or job.output_path is None:
            raise HTTPException(status_code=409, detail="Graph is not ready.")
        output_path = job.output_path
    if not output_path.is_file() or WEB_OUTPUT_ROOT not in output_path.parents:
        raise HTTPException(status_code=404, detail="Generated graph not found.")
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=output_path.name,
        headers={"Cache-Control": "private, max-age=86400"},
    )


def main() -> None:
    """Launch the single-process ASGI server for the website and API."""
    import uvicorn

    host = os.environ.get("VELOX_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", os.environ.get("VELOX_WEB_PORT", "8000")))
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
