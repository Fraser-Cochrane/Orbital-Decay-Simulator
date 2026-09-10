"""Coordinate validated requests, model execution, caching, and plotting."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

from . import model
from .config import SimulationRequest, SpacecraftParameters, VELOX_C1_DEFAULTS
from .plotting import plot_decay_rates, plot_solar_cycle_phase_surface


# File rendering is performed by the worker thread; no Matplotlib window is
# created there. The finished image can then be opened with the system viewer.
matplotlib.use("Agg")

ProgressCallback = Callable[[float, str], None]

# The scientific profile is the default. The opt-in demonstration profile is
# intentionally coarse so the GUI, propagation, plotting, and cache paths can
# be demonstrated within a few seconds.
QUICK_DEMO_MONTE_CARLO_RUNS = 8
QUICK_DEMO_PROPAGATION_STEP = 30.0 * model.SECONDS_PER_DAY


@dataclass(frozen=True, slots=True)
class RunResult:
    """Metadata returned to the GUI after a graph has been generated."""

    output_path: Path
    plot_kind: str
    message: str
    quick_demo: bool = False
    cache_hits: int = 0
    cache_trials: int = 0


def _report(
    callback: ProgressCallback | None,
    percentage: float,
    message: str,
) -> None:
    """Send a clipped progress value to a callback when one is registered."""
    if callback is not None:
        callback(float(np.clip(percentage, 0.0, 100.0)), message)


@contextmanager
def _selected_altitudes(altitudes_km: tuple[float, ...]) -> Iterator[np.ndarray]:
    """Select the requested model grid for one non-concurrent GUI task."""
    previous = model.SAMPLE_HEIGHTS_KM
    selected = np.asarray(altitudes_km, dtype=float)
    model.SAMPLE_HEIGHTS_KM = selected
    try:
        yield selected
    finally:
        model.SAMPLE_HEIGHTS_KM = previous


@contextmanager
def _simulation_profile(quick_demo: bool) -> Iterator[None]:
    """Temporarily select the isolated, explicitly approximate demo profile."""
    previous_runs = model.MONTE_CARLO_RUNS
    previous_step = model.PROPAGATION_STEP
    if quick_demo:
        model.MONTE_CARLO_RUNS = QUICK_DEMO_MONTE_CARLO_RUNS
        model.PROPAGATION_STEP = QUICK_DEMO_PROPAGATION_STEP
    try:
        yield
    finally:
        model.MONTE_CARLO_RUNS = previous_runs
        model.PROPAGATION_STEP = previous_step


@contextmanager
def _spacecraft_profile(parameters: SpacecraftParameters) -> Iterator[None]:
    """Temporarily expose validated spacecraft values to the model engine."""
    previous = {
        "SATELLITE_MASS": model.SATELLITE_MASS,
        "CROSS_SECTIONAL_AREA": model.CROSS_SECTIONAL_AREA,
        "VELOX_BODY_DIMENSIONS": model.VELOX_BODY_DIMENSIONS,
        "SOLAR_RADIATION_AREA": model.SOLAR_RADIATION_AREA,
        "SOLAR_REFLECTION_FACTOR": model.SOLAR_REFLECTION_FACTOR,
        "SOLAR_RADIATION_COEFFICIENT": model.SOLAR_RADIATION_COEFFICIENT,
        "ECCENTRICITY": model.ECCENTRICITY,
        "INCLINATION_DEG": model.INCLINATION_DEG,
        "RAAN_DEG": model.RAAN_DEG,
        "ARGUMENT_OF_PERIGEE_DEG": model.ARGUMENT_OF_PERIGEE_DEG,
    }
    model.SATELLITE_MASS = parameters.mass_kg
    model.CROSS_SECTIONAL_AREA = parameters.aerodynamic_area_m2
    model.VELOX_BODY_DIMENSIONS = np.asarray(parameters.body_dimensions_m)
    model.SOLAR_RADIATION_AREA = parameters.solar_radiation_area_m2
    model.SOLAR_REFLECTION_FACTOR = parameters.solar_reflection_factor
    model.SOLAR_RADIATION_COEFFICIENT = 1.0 + parameters.solar_reflection_factor
    model.ECCENTRICITY = parameters.eccentricity
    model.INCLINATION_DEG = parameters.inclination_deg
    model.RAAN_DEG = parameters.raan_deg
    model.ARGUMENT_OF_PERIGEE_DEG = parameters.argument_of_perigee_deg
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(model, name, value)


def heatmap_phase_epochs(start_epoch: np.datetime64 | str) -> np.ndarray:
    """Map a selected start epoch across the Cycle 25 ascent duration."""
    start_epoch = model.parse_launch_epoch(str(start_epoch))
    interval_seconds = int(
        (
            model.SOLAR_CYCLE_MAXIMUM_EPOCH
            - model.SOLAR_CYCLE_MINIMUM_EPOCH
        ).astype("timedelta64[s]").astype(np.int64)
    )
    offsets = np.rint(
        model.SOLAR_CYCLE_PHASE_PERCENTAGES * interval_seconds / 100.0
    ).astype("timedelta64[s]")
    return start_epoch + offsets


def _spacecraft_label(parameters: SpacecraftParameters) -> str:
    """Return the graph label associated with a spacecraft configuration."""
    return "VELOX-C1" if parameters == VELOX_C1_DEFAULTS else "Configured spacecraft"


def _model_progress(
    callback: ProgressCallback | None,
    start: float,
    span: float,
    message: str,
) -> Callable[[int, int], None]:
    """Translate model step counts into a bounded GUI progress interval."""
    last_integer = -1

    def update(completed: int, total: int) -> None:
        """Report integer progress changes while always emitting completion."""
        nonlocal last_integer
        fraction = completed / max(total, 1)
        percentage = start + span * fraction
        integer = int(percentage)
        if integer != last_integer or completed == total:
            last_integer = integer
            _report(callback, percentage, message)

    return update


def _run_model_with_cache_status(
    launch_epoch: np.datetime64,
    use_cache: bool,
    simulation_duration: float,
    progress_callback: Callable[[int, int], None],
) -> tuple[np.ndarray, bool]:
    """Run the model and report whether an exact verified cache was reused."""
    if use_cache:
        cached = model.load_cached_decay_rates(launch_epoch, simulation_duration)
        if cached is not None:
            progress_callback(1, 1)
            return cached, True

    rates = model.run_monte_carlo(
        launch_epoch,
        use_cache=use_cache,
        simulation_duration=simulation_duration,
        verbose=False,
        progress_callback=progress_callback,
    )
    return rates, False


def _completion_message(
    request: SimulationRequest,
    description: str,
    cache_hits: int,
    cache_trials: int,
) -> str:
    """Describe completion and exact-cache use for the selected run profile."""
    if not request.quick_demo:
        return description
    if not request.use_cache:
        return f"Quick demo complete without caching. {description}"
    if cache_hits == cache_trials:
        return f"Quick demo loaded entirely from exact cache. {description}"
    if cache_hits:
        return (
            f"Quick demo complete: {cache_hits}/{cache_trials} trials loaded "
            f"from exact cache. {description}"
        )
    return (
        "Quick demo computed and cached. Run it again to demonstrate the "
        f"exact cache. {description}"
    )


def generate_line_graph(
    request: SimulationRequest,
    output_directory: Path,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Run a one-year propagation and save its altitude line graph."""
    output_directory.mkdir(parents=True, exist_ok=True)
    launch_epoch = model.parse_launch_epoch(request.launch_date)
    launch_day = np.datetime_as_string(launch_epoch, unit="D")
    spacecraft_suffix = (
        "" if request.spacecraft == VELOX_C1_DEFAULTS else "_configured"
    )
    demo_suffix = "_quick_demo" if request.quick_demo else ""
    output_path = output_directory / (
        f"orbital_decay_rate_{launch_day}{spacecraft_suffix}{demo_suffix}.png"
    )

    with (
        _simulation_profile(request.quick_demo),
        _spacecraft_profile(request.spacecraft),
        _selected_altitudes(request.altitudes_km) as heights,
    ):
        preparation = (
            "Preparing approximate quick demo"
            if request.quick_demo
            else "Preparing the one-year propagation"
        )
        _report(progress, 0.0, preparation)
        decay_rates, cache_hit = _run_model_with_cache_status(
            launch_epoch=launch_epoch,
            use_cache=request.use_cache,
            simulation_duration=model.SIMULATION_DURATION,
            progress_callback=_model_progress(
                progress,
                2.0,
                93.0,
                (
                    "Running coarse approximate propagation"
                    if request.quick_demo
                    else "Propagating atmosphere, drag, and vector SRP"
                ),
            ),
        )
        _report(progress, 96.0, "Summarizing Monte Carlo uncertainty")
        mean, standard_deviation = model.summarize_decay_rates(decay_rates)
        plot_decay_rates(
            heights,
            mean,
            standard_deviation,
            launch_epoch,
            output_path=output_path,
            show=False,
            spacecraft_label=_spacecraft_label(request.spacecraft),
            quick_demo=request.quick_demo,
        )

    _report(progress, 100.0, "Line graph complete")
    return RunResult(
        output_path=output_path,
        plot_kind="line",
        message=_completion_message(
            request,
            f"One-year line graph generated from {launch_day} UTC.",
            int(cache_hit),
            1,
        ),
        quick_demo=request.quick_demo,
        cache_hits=int(cache_hit),
        cache_trials=1,
    )


def generate_heatmap(
    request: SimulationRequest,
    output_directory: Path,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Evaluate instantaneous solar-cycle epochs and save a heat map."""
    output_directory.mkdir(parents=True, exist_ok=True)
    interval_start = model.parse_launch_epoch(request.launch_date)
    interval_start_day = np.datetime_as_string(interval_start, unit="D")
    spacecraft_suffix = (
        "" if request.spacecraft == VELOX_C1_DEFAULTS else "_configured"
    )
    demo_suffix = "_quick_demo" if request.quick_demo else ""
    output_path = output_directory / (
        f"orbital_decay_phase_heatmap_{interval_start_day}"
        f"{spacecraft_suffix}{demo_suffix}.png"
    )
    phases = model.SOLAR_CYCLE_PHASE_PERCENTAGES
    phase_epochs = heatmap_phase_epochs(interval_start)
    interval_end = phase_epochs[-1]
    means: list[np.ndarray] = []
    standard_deviations: list[np.ndarray] = []
    cache_hits = 0

    with (
        _simulation_profile(request.quick_demo),
        _spacecraft_profile(request.spacecraft),
        _selected_altitudes(request.altitudes_km) as heights,
    ):
        preparation = (
            "Preparing approximate quick-demo phase samples"
            if request.quick_demo
            else "Preparing Solar Cycle 25 phase samples"
        )
        _report(progress, 0.0, preparation)
        for index, (phase, epoch) in enumerate(zip(phases, phase_epochs)):
            start = 2.0 + 92.0 * index / len(phase_epochs)
            span = 92.0 / len(phase_epochs)
            rates, cache_hit = _run_model_with_cache_status(
                launch_epoch=epoch,
                use_cache=request.use_cache,
                simulation_duration=model.INSTANTANEOUS_EVALUATION_DURATION,
                progress_callback=_model_progress(
                    progress,
                    start,
                    span,
                    f"Evaluating solar-cycle phase {phase:.0f}%",
                ),
            )
            cache_hits += int(cache_hit)
            mean, standard_deviation = model.summarize_decay_rates(rates)
            means.append(mean)
            standard_deviations.append(standard_deviation)

        _report(progress, 96.0, "Rendering heat map and uncertainty contours")
        plot_solar_cycle_phase_surface(
            phases,
            heights,
            np.stack(means),
            np.stack(standard_deviations),
            interval_start=interval_start,
            interval_end=interval_end,
            output_path=output_path,
            show=False,
            spacecraft_label=_spacecraft_label(request.spacecraft),
            quick_demo=request.quick_demo,
        )

    _report(progress, 100.0, "Heat map complete")
    return RunResult(
        output_path=output_path,
        plot_kind="heatmap",
        message=_completion_message(
            request,
            (
                "Instantaneous phase heat map generated for interval beginning "
                f"{interval_start_day}."
            ),
            cache_hits,
            len(phase_epochs),
        ),
        quick_demo=request.quick_demo,
        cache_hits=cache_hits,
        cache_trials=len(phase_epochs),
    )


def run_request(
    request: SimulationRequest,
    output_directory: Path,
    progress: ProgressCallback | None = None,
) -> RunResult:
    """Dispatch one validated request without changing model equations."""
    if request.plot_kind == "line":
        return generate_line_graph(request, output_directory, progress)
    return generate_heatmap(request, output_directory, progress)
