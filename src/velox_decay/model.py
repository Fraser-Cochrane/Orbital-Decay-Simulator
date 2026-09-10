"""Monte Carlo orbital decay prediction for the VELOX-C1 microsatellite.

The program samples initial altitudes from 500 km through 700 km and evaluates
NRLMSIS 2.1 at the UTC time and geodetic position of every orbit sample. Solar
F10.7, its 81-day average, and storm-time Ap provide the atmospheric drivers.

The atmosphere, drag, and uncertainty treatment combines:

* observed space-weather drivers without artificial perturbations, and
  data-estimated F10.7/Ap forecast-error dependence when forecasts are used;
* NRLMSIS 2.1 mass density, neutral composition, and temperature;
* central second-order NRLMSIS response surfaces for trajectory altitude and
  forecast-driver dispersion, including composition/temperature effects on Cd;
* temporally and spatially correlated log-density model discrepancy;
* a composition- and temperature-dependent Sentman free-molecular drag
  coefficient with atomic-oxygen adsorption and incomplete accommodation;
* persistent density and gas-surface model discrepancy plus time-correlated
  horizontal neutral-wind uncertainty;
* configured VELOX mass, geometry, material, wall temperature, ram-pointing
  attitude, and initial orbital elements;
* the published VELOX-C1 flight result as an independent diagnostic (not as
  an unjustified universal correction) and first-order J2 secular precession;
* Sun-directed vector solar-radiation pressure with inverse-square scaling,
  finite solar-disc Earth eclipses, and separate orbit averaging;
* the resulting nonlinear orbital response to atmospheric drag and SRP.

The plotted values are the Monte Carlo mean and +/- one sample standard
deviation. No fixed density-bias adjustment is applied.

For repeated or batched analyses, completed exact-match runs are cached. Use,
for example:

    python run_cli.py --start 2024-01-01 --no-show
    python run_cli.py --start 2022-01-01 \
        --start 2023-01-01 --start 2024-01-01 --workers 3 --no-show
    python run_cli.py --solar-cycle-comparison \
        --workers 3 --no-show
    python run_cli.py --solar-cycle-surface --no-show

The worker pipeline parallelizes independent start periods; it never splits a
single trajectory because every propagation interval depends on the preceding
orbital state. Pass --no-cache to force fresh calculations.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from math import pi
import os
from pathlib import Path

import numpy as np
import pymsis
from pymsis import Variable
from pymsis.utils import download_f107_ap, get_f107_ap


# Keep generated artifacts outside the importable package. The environment
# variable supports installed use while retaining a predictable repository
# default when launched through the supplied entry points.
_SOURCE_FILE = (
    Path(__file__).resolve() if "__file__" in globals() else None
)
RUNTIME_DIRECTORY = Path(
    os.environ.get("VELOX_DECAY_OUTPUT_DIR", Path.cwd() / "outputs")
).resolve()
RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Spacecraft and Earth constants (SI units unless stated otherwise)

CROSS_SECTIONAL_AREA = 0.52  # m^2
SATELLITE_MASS = 123.0  # kg

# VELOX-C1 is a rectangular microsatellite rather than a CubeSat. The published
# orbit-decay study gives 0.615 x 0.608 x 0.848 m and uses a broadside reference
# area of approximately 0.52 m^2. The model assumes that this broad face remains
# normal to the ram direction; attitude telemetry was not published with the
# decay validation data.
VELOX_BODY_DIMENSIONS = np.array((0.615, 0.608, 0.848))  # m
SURFACE_TEMPERATURE = 300.0  # K; common thermospheric-drag assumption

# Diffuse reflection with incomplete accommodation (DRIA). The Langmuir model
# represents the fraction of the surface covered by adsorbed atomic oxygen.
# Goodman accommodation represents the uncovered surface. VELOX-C1's detailed
# external material map is unavailable; 28 g/mol is a transparent Al/Si proxy
# for its structure and solar-cell exterior (their molar masses are similar).
LANGMUIR_DRIA_CONSTANT = 1.44e6  # Pa^-1, Walker et al. DRIA fit
# Surface material parameters are treated as exact scenario inputs.
GOODMAN_SUBSTRATE_COEFFICIENT = 3.0
EFFECTIVE_SURFACE_MOLAR_MASS = 28.0e-3  # kg mol^-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J K^-1

# Low and Chia (doi:10.26077/bffw-p652) report 2.566 km of GPS-observed decay
# and 2.444 km from their reference model over 331 days (2015-12-16 through
# 2016-11-11). The ratio is reported for validation only. Applying it directly
# to the physical Sentman coefficient would confound atmospheric-density error,
# attitude, area, solar-pressure treatment, and drag coefficient.
VELOX_OBSERVED_DECAY_M = 2_566.0
VELOX_REFERENCE_MODEL_DECAY_M = 2_444.0
PUBLISHED_VALIDATION_RATIO = (
    VELOX_OBSERVED_DECAY_M / VELOX_REFERENCE_MODEL_DECAY_M
)
# Species provided by PyMSIS and their molar masses. A mass-fraction-weighted
# mixture coefficient is required because each species has a different thermal
# speed. Anomalous oxygen has the same molar mass as atomic oxygen.
MSIS_DRAG_SPECIES = (
    Variable.N2,
    Variable.O2,
    Variable.O,
    Variable.HE,
    Variable.H,
    Variable.AR,
    Variable.N,
    Variable.ANOMALOUS_O,
    Variable.NO,
)
SPECIES_MOLAR_MASS = np.array(
    (28.0134, 31.9988, 15.999, 4.002602, 1.00794, 39.948,
     14.0067, 15.999, 30.0061)
) * 1e-3  # kg mol^-1
UNIVERSAL_GAS_CONSTANT = 8.31446261815324  # J mol^-1 K^-1

EARTH_MU = 3.986004418e14  # m^3 s^-2
EARTH_EQUATORIAL_RADIUS = 6_378_137.0  # m, WGS84
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84
EARTH_POLAR_RADIUS = EARTH_EQUATORIAL_RADIUS * (1.0 - EARTH_FLATTENING)
EARTH_ECCENTRICITY_SQUARED = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
EARTH_SECOND_ECCENTRICITY_SQUARED = (
    EARTH_EQUATORIAL_RADIUS**2 - EARTH_POLAR_RADIUS**2
) / EARTH_POLAR_RADIUS**2
EARTH_SIDEREAL_DAY = 86_164.0905  # s
EARTH_ROTATION_RATE = 2.0 * pi / EARTH_SIDEREAL_DAY  # rad s^-1
EARTH_J2 = 1.08262668e-3

# Direct solar-radiation pressure. Low and Chia use a reflection factor of 0.5
# for VELOX-C1; its equivalent cannonball radiation coefficient is therefore
# 1.5. Unlike their scalar decay term, the acceleration below is resolved along
# the instantaneous Sun-spacecraft line and is zero in the Earth's shadow.
# The optical area is deliberately separate from the aerodynamic reference
# area even though both published VELOX values are currently 0.52 m^2.
ASTRONOMICAL_UNIT = 149_597_870_700.0  # m, IAU 2012 exact definition
SUN_RADIUS = 695_700_000.0  # m, nominal photospheric radius
SOLAR_RADIATION_PRESSURE_AT_1_AU = 4.56e-6  # N m^-2
SOLAR_REFLECTION_FACTOR = 0.5
SOLAR_RADIATION_COEFFICIENT = 1.0 + SOLAR_REFLECTION_FACTOR
SOLAR_RADIATION_AREA = 0.52  # m^2, effective Sun-normal cannonball area


# ---------------------------------------------------------------------------
# Orbit configuration
#
# The published VELOX-C1 validation describes a near-circular, 15-degree orbit.
# RAAN and argument of perigee remain exposed assumptions because their values
# were not supplied in the paper's model summary.

ECCENTRICITY = 0.0009024
INCLINATION_DEG = 15.0
RAAN_DEG = 0.0
ARGUMENT_OF_PERIGEE_DEG = 0.0

SAMPLE_HEIGHTS_KM = np.arange(500.0, 701.0, 10.0)
# Solar Cycle 25 comparison epochs. NASA/NOAA place the minimum in December
# 2019 and announced the maximum period on 2024-10-15. The midpoint below is
# explicitly the temporal midpoint between those reference dates, not a claim
# that solar activity evolves linearly between them.
SOLAR_CYCLE_COMPARISON_EPOCHS = (
    ("Maximum", "2024-10-15", "tab:red"),
    ("Midpoint", "2022-05-09", "#d4a900"),
    ("Minimum", "2019-12-01", "tab:green"),
)
# The three-dimensional surface samples the observed ascending interval more
# densely than the three comparison curves. Percentage is therefore explicitly
# normalized from the confirmed Cycle 25 minimum (0%) to the NOAA/NASA maximum
# reference date (100%); it is not a prediction of the next solar minimum.
SOLAR_CYCLE_MINIMUM_EPOCH = np.datetime64("2019-12-01T00:00:00", "s")
SOLAR_CYCLE_MAXIMUM_EPOCH = np.datetime64("2024-10-15T00:00:00", "s")
SOLAR_CYCLE_PHASE_PERCENTAGES = np.linspace(0.0, 100.0, 21)
INSTANTANEOUS_EVALUATION_DURATION = 1.0  # s; one epoch-local force evaluation
ORBIT_SAMPLES_PER_STEP = 24
# Eclipse entry and exit are much sharper than the atmospheric variation. SRP
# is consequently integrated on a separate deterministic orbit grid rather
# than expanding the full 200-realization atmosphere cube to this resolution.
SRP_ORBIT_SAMPLES = 720
PROPAGATION_STEP = 3.0 * 3_600.0  # s; matches storm-time Ap cadence
SECONDS_PER_DAY = 86_400.0
SIMULATION_DURATION = 365.2475 * SECONDS_PER_DAY
REENTRY_ALTITUDE = 120_000.0  # m


# ---------------------------------------------------------------------------
# Monte Carlo settings

MONTE_CARLO_RUNS = 200
RANDOM_SEED = 42

# HWM14 validation reports approximately 37 m/s wind uncertainty. HWM14 is not
# available in this runtime, so the unresolved horizontal wind about rigid
# corotation is propagated as a zero-mean Gauss-Markov error. Its three-hour
# correlation time is an exposed assumption, not a claimed HWM14 parameter.
NEUTRAL_WIND_SIGMA_M_PER_S = 37.0
NEUTRAL_WIND_TIME_CORRELATION = 3.0 * 3_600.0

# One-sigma space-weather forecast/input uncertainties. These are deliberately
# exposed rather than hidden in the density calculation and should be replaced
# with covariance estimates from the chosen operational forecast product.
F107_SIGMA = 8.0  # solar flux units
LOG_AP_PLUS_ONE_SIGMA = 0.35

# The innovation correlation between [F10.7, log(Ap + 1)] is estimated from
# observed, pre-launch CelesTrak data at runtime. F10.7a is derived from each
# sampled F10.7 history rather than assigned an independent random process.
SPACE_WEATHER_CORRELATION_TIMES = np.array(
    (27.0, 1.0)
) * SECONDS_PER_DAY
CORRELATION_TRAINING_YEARS = 11.0
CORRELATION_BLOCK_LENGTH_DAYS = 27
CORRELATION_BOOTSTRAP_SAMPLES = 500
CORRELATION_WINSOR_QUANTILE = 0.01
CORRELATION_SIGNIFICANCE_LEVEL = 0.05
CORRELATION_RANDOM_SEED = 8_641

# Residual model-discrepancy assumptions. The coefficient of variation grows
# with altitude and geomagnetic disturbance. The residual is correlated in
# time and between nearby altitude samples.
MODEL_ERROR_CV_AT_500_KM = 0.14
MODEL_ERROR_CV_PER_100_KM = 0.03
MODEL_ERROR_STORM_SCALE_PER_AP = 0.005
MODEL_ERROR_TIME_CORRELATION = 6.0 * 3_600.0
MODEL_ERROR_ALTITUDE_CORRELATION_KM = 100.0
# A trajectory-persistent density scale captures model bias that cannot average
# away like short-timescale weather. Ten percent is consistent with the order
# of the independent CHAMP bias reported for NRLMSIS 2.0; it is kept separate
# from the time-varying residual above.
DENSITY_SYSTEMATIC_CV = 0.10
# Published comparisons of plausible gas-surface interaction formulations show
# roughly 2-15% Cd differences near 500 km depending on solar conditions. An
# 8% persistent lognormal term represents this structural model discrepancy.
DRAG_COEFFICIENT_SYSTEMATIC_CV = 0.08

# The initial-altitude grid and launch epoch are scenario values.
# All spacecraft and initial-orbit parameters are treated as exact. Fundamental
# constants and WGS84 parameters have uncertainties many orders below the
# atmosphere/GSI terms. Numerical error is convergence-tested, not randomized.

# Symmetric finite differences used to propagate input uncertainty through the
# nonlinear NRLMSIS coupling. Gradients and curvatures are evaluated in log
# density, preserving positivity and the nearly exponential altitude response.
F107_PERTURBATION = 1.0
F107A_PERTURBATION = 1.0
LOG_AP_PLUS_ONE_PERTURBATION = 0.05
ALTITUDE_PERTURBATION_KM = 1.0

# NRLMSIS 2.1 retains the 2.0 mass-density formulation and adds nitric oxide.
# NRLMSISE-00 is not mixed into the ensemble: treating a superseded model as
# equally credible would add spread without a defensible statistical meaning.
MSIS_VERSION = 2.1

# Exact completed-run cache. Cached values are reused only when the launch
# epoch, numerical configuration, source code, PyMSIS version, and CelesTrak
# driver file all match. It therefore accelerates repeated analyses without
# approximating nearby epochs or silently reusing stale atmospheric data.
ENABLE_RESULT_CACHE = True
CACHE_SCHEMA_VERSION = 6
RESULT_CACHE_DIRECTORY = RUNTIME_DIRECTORY / ".orbital_decay_cache"
_CACHE_BASE_DIGEST: str | None = None


def space_weather_file_path() -> Path:
    """Return the configured CelesTrak driver file, downloading if needed.

    PyMSIS 0.12 ships ``SW-All.csv`` in the package, while PyMSIS 0.13
    downloads it on first use. Resolving the file through this helper keeps
    both layouts working and honours an explicitly configured data file.
    """
    configured_path = os.environ.get("PYMSIS_SPACE_WEATHER_FILE")
    if configured_path:
        space_weather_path = Path(configured_path).expanduser().resolve()
        if not space_weather_path.is_file():
            raise RuntimeError(
                "The configured PyMSIS space-weather file is unavailable: "
                f"{space_weather_path}"
            )
        return space_weather_path

    space_weather_path = Path(pymsis.__file__).with_name("SW-All.csv")
    if not space_weather_path.is_file():
        try:
            download_f107_ap()
        except OSError as error:
            raise RuntimeError(
                "PyMSIS could not download the CelesTrak space-weather file. "
                "Check the internet connection and try again."
            ) from error
    if not space_weather_path.is_file():
        raise RuntimeError("The PyMSIS space-weather file is unavailable.")
    return space_weather_path


def parse_launch_epoch(text: str) -> np.datetime64:
    """Parse an ISO-8601 launch epoch and normalize it to second precision."""
    try:
        epoch = np.datetime64(text.strip(), "s")
    except ValueError as error:
        raise ValueError(
            "Launch epoch must use ISO-8601 format, for example "
            "2024-01-01T00:00:00."
        ) from error

    if np.isnat(epoch):
        raise ValueError("Launch epoch cannot be NaT.")
    return epoch


def cache_base_digest() -> str:
    """Return a process-local digest of code, model, and driver data."""
    global _CACHE_BASE_DIGEST
    if _CACHE_BASE_DIGEST is not None:
        return _CACHE_BASE_DIGEST

    digest = hashlib.sha256()
    digest.update(f"schema={CACHE_SCHEMA_VERSION}".encode())
    if _SOURCE_FILE is not None and _SOURCE_FILE.exists():
        digest.update(_SOURCE_FILE.read_bytes())
    else:
        # Notebook cells have no stable source file to hash. The cache schema
        # and the explicit runtime configuration below provide invalidation;
        # increment CACHE_SCHEMA_VERSION after editing model equations in-place.
        digest.update(b"interactive-notebook-source")
    digest.update(str(getattr(pymsis, "__version__", "unknown")).encode())
    space_weather_path = space_weather_file_path()
    with space_weather_path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    _CACHE_BASE_DIGEST = digest.hexdigest()
    return _CACHE_BASE_DIGEST


def result_cache_key(
    launch_epoch: np.datetime64,
    simulation_duration: float = SIMULATION_DURATION,
) -> str:
    """Return an exact cache key for one complete Monte Carlo propagation."""
    runtime_configuration = {
        "base_digest": cache_base_digest(),
        "launch_epoch": np.datetime_as_string(launch_epoch, unit="s"),
        "sample_heights_km": SAMPLE_HEIGHTS_KM.tolist(),
        "orbit_samples_per_step": ORBIT_SAMPLES_PER_STEP,
        "srp_orbit_samples": SRP_ORBIT_SAMPLES,
        "propagation_step_s": PROPAGATION_STEP,
        "simulation_duration_s": simulation_duration,
        "monte_carlo_runs": MONTE_CARLO_RUNS,
        "random_seed": RANDOM_SEED,
        "mass_kg": SATELLITE_MASS,
        "reference_area_m2": CROSS_SECTIONAL_AREA,
        "solar_radiation_area_m2": SOLAR_RADIATION_AREA,
        "solar_radiation_coefficient": SOLAR_RADIATION_COEFFICIENT,
        "solar_pressure_at_1_au_n_m2": (
            SOLAR_RADIATION_PRESSURE_AT_1_AU
        ),
        "astronomical_unit_m": ASTRONOMICAL_UNIT,
        "sun_radius_m": SUN_RADIUS,
        "dimensions_m": VELOX_BODY_DIMENSIONS.tolist(),
        "surface_temperature_k": SURFACE_TEMPERATURE,
        "surface_molar_mass": EFFECTIVE_SURFACE_MOLAR_MASS,
        "langmuir_dria_constant_pa_inverse": LANGMUIR_DRIA_CONSTANT,
        "eccentricity": ECCENTRICITY,
        "inclination_deg": INCLINATION_DEG,
        "raan_deg": RAAN_DEG,
        "argument_of_perigee_deg": ARGUMENT_OF_PERIGEE_DEG,
        "msis_version": MSIS_VERSION,
        "goodman_coefficient": GOODMAN_SUBSTRATE_COEFFICIENT,
        "f107_sigma": F107_SIGMA,
        "log_ap_sigma": LOG_AP_PLUS_ONE_SIGMA,
        "space_weather_correlation_times_s": (
            SPACE_WEATHER_CORRELATION_TIMES.tolist()
        ),
        "correlation_training_years": CORRELATION_TRAINING_YEARS,
        "correlation_block_length_days": CORRELATION_BLOCK_LENGTH_DAYS,
        "correlation_bootstrap_samples": CORRELATION_BOOTSTRAP_SAMPLES,
        "correlation_winsor_quantile": CORRELATION_WINSOR_QUANTILE,
        "correlation_significance_level": (
            CORRELATION_SIGNIFICANCE_LEVEL
        ),
        "correlation_random_seed": CORRELATION_RANDOM_SEED,
        "model_error_cv_500": MODEL_ERROR_CV_AT_500_KM,
        "model_error_cv_per_100": MODEL_ERROR_CV_PER_100_KM,
        "model_error_storm_scale": MODEL_ERROR_STORM_SCALE_PER_AP,
        "model_error_time_correlation_s": MODEL_ERROR_TIME_CORRELATION,
        "model_error_altitude_correlation_km": (
            MODEL_ERROR_ALTITUDE_CORRELATION_KM
        ),
        "density_systematic_cv": DENSITY_SYSTEMATIC_CV,
        "drag_systematic_cv": DRAG_COEFFICIENT_SYSTEMATIC_CV,
        "neutral_wind_sigma": NEUTRAL_WIND_SIGMA_M_PER_S,
        "neutral_wind_correlation_s": NEUTRAL_WIND_TIME_CORRELATION,
        "f107_perturbation": F107_PERTURBATION,
        "f107a_perturbation": F107A_PERTURBATION,
        "log_ap_perturbation": LOG_AP_PLUS_ONE_PERTURBATION,
        "altitude_perturbation_km": ALTITUDE_PERTURBATION_KM,
        "reentry_altitude_m": REENTRY_ALTITUDE,
    }
    serialized = json.dumps(
        runtime_configuration, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def load_cached_decay_rates(
    launch_epoch: np.datetime64,
    simulation_duration: float = SIMULATION_DURATION,
) -> np.ndarray | None:
    """Load a verified exact-match result, or return None on a cache miss."""
    cache_key = result_cache_key(launch_epoch, simulation_duration)
    cache_path = RESULT_CACHE_DIRECTORY / f"{cache_key}.npz"
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            stored_key = str(cached["cache_key"])
            decay_rates = cached["decay_rates"]
    except (OSError, ValueError, KeyError):
        return None

    expected_shape = (MONTE_CARLO_RUNS, len(SAMPLE_HEIGHTS_KM))
    if (
        stored_key != cache_key
        or decay_rates.shape != expected_shape
        or not np.all(np.isfinite(decay_rates))
    ):
        return None
    print(f"Loaded exact cached result from {cache_path}")
    return decay_rates


def save_cached_decay_rates(
    launch_epoch: np.datetime64,
    decay_rates: np.ndarray,
    simulation_duration: float = SIMULATION_DURATION,
) -> None:
    """Atomically store one completed propagation for exact future reuse."""
    cache_key = result_cache_key(launch_epoch, simulation_duration)
    RESULT_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    cache_path = RESULT_CACHE_DIRECTORY / f"{cache_key}.npz"
    temporary_path = RESULT_CACHE_DIRECTORY / (
        f".{cache_key}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(
        temporary_path,
        cache_key=np.asarray(cache_key),
        decay_rates=decay_rates,
    )
    os.replace(temporary_path, cache_path)
    print(f"Cached completed result at {cache_path}")


def build_time_grid(
    launch_epoch: np.datetime64,
    simulation_duration: float = SIMULATION_DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interval midpoints and their possibly shortened step lengths."""
    if simulation_duration <= 0.0:
        raise ValueError("simulation_duration must be positive")
    number_of_steps = int(np.ceil(simulation_duration / PROPAGATION_STEP))
    offsets = np.arange(number_of_steps, dtype=np.int64) * int(PROPAGATION_STEP)
    steps = np.full(number_of_steps, PROPAGATION_STEP)
    steps[-1] = simulation_duration - PROPAGATION_STEP * (number_of_steps - 1)
    midpoint_offsets = np.rint(offsets + 0.5 * steps).astype("timedelta64[s]")
    dates = launch_epoch + midpoint_offsets
    return dates, steps


def load_nominal_space_weather(
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load historical/predicted F10.7 and storm-time Ap data for MSIS."""
    try:
        f107, f107a, ap = get_f107_ap(dates)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "PyMSIS could not obtain space-weather data for the requested "
            "dates. Check the internet connection and ensure the launch year "
            "is covered by the CelesTrak space-weather file."
        ) from error

    if not (
        np.all(np.isfinite(f107))
        and np.all(np.isfinite(f107a))
        and np.all(np.isfinite(ap))
    ):
        raise RuntimeError("The selected space-weather interval contains gaps.")
    return f107, f107a, ap


def observed_space_weather_masks(
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate observation masks for F10.7 and the Ap history."""
    space_weather_path = space_weather_file_path()

    requested_days = dates.astype("datetime64[D]")
    earliest_day = np.min(requested_days) - np.timedelta64(3, "D")
    latest_day = np.max(requested_days)
    data_types: dict[str, str] = {}
    with space_weather_path.open(newline="") as file:
        for row in csv.DictReader(file):
            date_text = row.get("DATE", "")
            try:
                day = np.datetime64(date_text, "D")
            except ValueError:
                continue
            if earliest_day <= day <= latest_day:
                data_types[date_text] = row.get("F10.7_DATA_TYPE", "")

    f107_observed = np.empty(len(requested_days), dtype=bool)
    ap_observed = np.empty(len(requested_days), dtype=bool)
    ap_history_offsets = np.arange(-3, 1).astype("timedelta64[D]")
    for index, day in enumerate(requested_days):
        # PyMSIS uses the previous day's F10.7. Its centred 81-day average is
        # not independently perturbed below: it is derived from the sampled
        # F10.7 sequence, so one interpolated day is diluted by the averaging
        # operation rather than turning an entire 81-day interval into a
        # forecast.
        f107_observed[index] = (
            data_types.get(str(day - np.timedelta64(1, "D"))) == "OBS"
        )
        # CelesTrak has no separate Ap provenance flag. Historical rows whose
        # F10.7 is marked INT still contain measured Ap; PRD rows do not. The
        # four-day test conservatively covers every component of MSIS's
        # current-to-57-hour Ap vector.
        ap_history = day + ap_history_offsets
        ap_observed[index] = all(
            data_types.get(str(history_day)) in {"OBS", "INT"}
            for history_day in ap_history
        )
    return f107_observed, ap_observed


def load_observed_driver_history(
    launch_epoch: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an observed pre-launch F10.7 and Ap training window."""
    space_weather_path = space_weather_file_path()

    launch_day = launch_epoch.astype("datetime64[D]")
    training_days = int(round(CORRELATION_TRAINING_YEARS * 365.2425))
    training_start = launch_day - np.timedelta64(training_days, "D")

    dates = []
    f107_values = []
    log_ap_values = []
    with space_weather_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get("F10.7_DATA_TYPE") != "OBS":
                continue
            try:
                date = np.datetime64(row["DATE"], "D")
                f107 = float(row["F10.7_OBS"])
                ap = float(row["AP_AVG"])
            except (TypeError, ValueError):
                continue
            if training_start <= date < launch_day:
                dates.append(date)
                f107_values.append(f107)
                log_ap_values.append(np.log1p(max(ap, 0.0)))

    if len(dates) < 3 * 365:
        raise RuntimeError(
            "At least three years of observed pre-launch space-weather data "
            "are required to estimate driver correlations."
        )
    return (
        np.asarray(dates),
        np.asarray(f107_values),
        np.asarray(log_ap_values),
    )


def nearest_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric estimate to a positive-semidefinite correlation."""
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    projected = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
    scales = np.sqrt(np.diag(projected))
    projected /= np.outer(scales, scales)
    np.fill_diagonal(projected, 1.0)
    return projected


def estimate_driver_innovation_correlation(
    launch_epoch: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Estimate F10.7/Ap innovation correlation with block-bootstrap CIs."""
    dates, f107, log_ap = load_observed_driver_history(launch_epoch)
    levels = np.column_stack((f107, log_ap))

    # Use only differences between consecutive days. First differences remove
    # solar-cycle trends and target shocks that drive the stochastic process.
    consecutive = np.diff(dates).astype("timedelta64[D]").astype(int) == 1
    innovations = np.diff(levels, axis=0)[consecutive]

    # Winsorization limits leverage from isolated radio bursts and extreme
    # storms without discarding those dates entirely.
    lower_limits = np.quantile(
        innovations, CORRELATION_WINSOR_QUANTILE, axis=0
    )
    upper_limits = np.quantile(
        innovations, 1.0 - CORRELATION_WINSOR_QUANTILE, axis=0
    )
    innovations = np.clip(innovations, lower_limits, upper_limits)
    sample_correlation = np.corrcoef(innovations, rowvar=False)

    # A moving-block bootstrap preserves short-range serial dependence. If a
    # cross-correlation is not distinguishable from zero, set it to zero to
    # avoid injecting unsupported coupling into the Monte Carlo ensemble.
    generator = np.random.default_rng(CORRELATION_RANDOM_SEED)
    block_length = min(CORRELATION_BLOCK_LENGTH_DAYS, len(innovations))
    blocks_needed = int(np.ceil(len(innovations) / block_length))
    maximum_start = len(innovations) - block_length
    bootstrap_correlations = np.empty(
        (CORRELATION_BOOTSTRAP_SAMPLES, 2, 2)
    )
    for sample_index in range(CORRELATION_BOOTSTRAP_SAMPLES):
        starts = generator.integers(
            0, maximum_start + 1, size=blocks_needed
        )
        sample = np.concatenate(
            [
                innovations[start : start + block_length]
                for start in starts
            ],
            axis=0,
        )[: len(innovations)]
        bootstrap_correlations[sample_index] = np.corrcoef(
            sample, rowvar=False
        )

    tail_probability = CORRELATION_SIGNIFICANCE_LEVEL / 2.0
    lower = np.quantile(bootstrap_correlations, tail_probability, axis=0)
    upper = np.quantile(
        bootstrap_correlations, 1.0 - tail_probability, axis=0
    )
    estimated = sample_correlation.copy()
    for row in range(estimated.shape[0]):
        for column in range(row):
            if lower[row, column] <= 0.0 <= upper[row, column]:
                estimated[row, column] = 0.0
                estimated[column, row] = 0.0

    return nearest_correlation_matrix(estimated), lower, upper, len(innovations)


def print_driver_correlation_estimate(
    correlation: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sample_size: int,
) -> None:
    """Report the data-derived table and its off-diagonal confidence interval."""
    print(
        f"Estimated innovation correlation from {sample_size} pre-launch days "
        f"([F10.7, log(Ap+1)]):"
    )
    print(np.array2string(correlation, precision=3, suppress_small=True))
    print(
        "Moving-block bootstrap 95% interval for the cross-correlation: "
        f"[{lower[0, 1]:.3f}, {upper[0, 1]:.3f}]"
    )


def solve_true_anomaly(
    mean_anomaly: np.ndarray,
    eccentricity: np.ndarray | float = ECCENTRICITY,
) -> np.ndarray:
    """Solve Kepler's equation and return true anomaly in radians."""
    mean_anomaly, eccentricity = np.broadcast_arrays(
        mean_anomaly, eccentricity
    )
    # A second-order elliptic expansion is already extremely close for the
    # VELOX-C1 eccentricity. Newton iterations then stop at machine precision
    # instead of always performing eight full passes over the Monte Carlo cube.
    eccentric_anomaly = (
        mean_anomaly
        + eccentricity * np.sin(mean_anomaly)
        + 0.5 * eccentricity**2 * np.sin(2.0 * mean_anomaly)
    )
    for _ in range(12):
        residual = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        correction = residual / derivative
        eccentric_anomaly -= correction
        if np.max(np.abs(correction)) < 2e-14:
            break

    numerator = np.sqrt(1.0 + eccentricity) * np.sin(
        eccentric_anomaly / 2.0
    )
    denominator = np.sqrt(1.0 - eccentricity) * np.cos(
        eccentric_anomaly / 2.0
    )
    return 2.0 * np.arctan2(numerator, denominator)


MEAN_ANOMALY_SAMPLES = (
    2.0
    * pi
    * (np.arange(ORBIT_SAMPLES_PER_STEP) + 0.5)
    / ORBIT_SAMPLES_PER_STEP
)
TRUE_ANOMALY_SAMPLES = solve_true_anomaly(MEAN_ANOMALY_SAMPLES)
SRP_MEAN_ANOMALY_SAMPLES = (
    2.0
    * pi
    * (np.arange(SRP_ORBIT_SAMPLES) + 0.5)
    / SRP_ORBIT_SAMPLES
)


def sun_position_eci(epoch: np.datetime64) -> np.ndarray:
    """Return the geocentric Sun position in equatorial inertial coordinates.

    This compact solar ephemeris retains the dominant equation-of-centre and
    obliquity terms. Its angular accuracy is ample for the approximately
    half-degree solar disc and is far below the uncertainty from VELOX-C1's
    unpublished optical facet map and attitude history.
    """
    unix_seconds = (
        epoch - np.datetime64("1970-01-01T00:00:00", "s")
    ) / np.timedelta64(1, "s")
    julian_date = float(unix_seconds) / SECONDS_PER_DAY + 2_440_587.5
    centuries = (julian_date - 2_451_545.0) / 36_525.0
    mean_longitude = np.deg2rad(
        (280.4606184 + 36_000.77005361 * centuries) % 360.0
    )
    mean_anomaly = np.deg2rad(
        (357.5277233 + 35_999.05034 * centuries) % 360.0
    )
    ecliptic_longitude = (
        mean_longitude
        + np.deg2rad(1.914666471) * np.sin(mean_anomaly)
        + np.deg2rad(0.019994643) * np.sin(2.0 * mean_anomaly)
    )
    obliquity = np.deg2rad(
        23.439291 - 0.0130042 * centuries
    )
    distance_au = (
        1.000140612
        - 0.016708617 * np.cos(mean_anomaly)
        - 0.000139589 * np.cos(2.0 * mean_anomaly)
    )
    direction = np.array(
        (
            np.cos(ecliptic_longitude),
            np.cos(obliquity) * np.sin(ecliptic_longitude),
            np.sin(obliquity) * np.sin(ecliptic_longitude),
        )
    )
    return ASTRONOMICAL_UNIT * distance_au * direction


def sunlight_fraction(
    position_eci: np.ndarray,
    sun_position: np.ndarray,
) -> np.ndarray:
    """Return the visible fraction of the finite solar disc in [0, 1]."""
    spacecraft_to_sun = sun_position - position_eci
    spacecraft_to_earth = -position_eci
    sun_distance = np.linalg.norm(spacecraft_to_sun, axis=-1)
    earth_distance = np.linalg.norm(spacecraft_to_earth, axis=-1)
    sun_direction = spacecraft_to_sun / sun_distance[..., None]
    earth_direction = spacecraft_to_earth / earth_distance[..., None]
    separation = np.arccos(
        np.clip(
            np.sum(sun_direction * earth_direction, axis=-1),
            -1.0,
            1.0,
        )
    )
    sun_angular_radius = np.arcsin(
        np.clip(SUN_RADIUS / sun_distance, 0.0, 1.0)
    )
    earth_angular_radius = np.arcsin(
        np.clip(EARTH_EQUATORIAL_RADIUS / earth_distance, 0.0, 1.0)
    )

    visible = np.ones_like(separation)
    fully_eclipsed = separation <= (
        earth_angular_radius - sun_angular_radius
    )
    visible[fully_eclipsed] = 0.0
    partial = (~fully_eclipsed) & (
        separation < earth_angular_radius + sun_angular_radius
    )
    if np.any(partial):
        centre_distance = separation[partial]
        solar_radius = sun_angular_radius[partial]
        terrestrial_radius = earth_angular_radius[partial]
        solar_angle = np.arccos(
            np.clip(
                (
                    centre_distance**2
                    + solar_radius**2
                    - terrestrial_radius**2
                )
                / (2.0 * centre_distance * solar_radius),
                -1.0,
                1.0,
            )
        )
        terrestrial_angle = np.arccos(
            np.clip(
                (
                    centre_distance**2
                    + terrestrial_radius**2
                    - solar_radius**2
                )
                / (2.0 * centre_distance * terrestrial_radius),
                -1.0,
                1.0,
            )
        )
        lens_term = np.sqrt(
            np.maximum(
                (
                    -centre_distance + solar_radius + terrestrial_radius
                )
                * (
                    centre_distance + solar_radius - terrestrial_radius
                )
                * (
                    centre_distance - solar_radius + terrestrial_radius
                )
                * (
                    centre_distance + solar_radius + terrestrial_radius
                ),
                0.0,
            )
        )
        overlap = (
            solar_radius**2 * solar_angle
            + terrestrial_radius**2 * terrestrial_angle
            - 0.5 * lens_term
        )
        visible[partial] = 1.0 - overlap / (pi * solar_radius**2)
    return np.clip(visible, 0.0, 1.0)


def orbital_state_eci(
    semi_major_axis: np.ndarray,
    true_anomaly: np.ndarray,
    raan: np.ndarray | float | None = None,
    argument_of_perigee: np.ndarray | float | None = None,
    eccentricity: np.ndarray | float = ECCENTRICITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position and velocity in an Earth-centred inertial frame."""
    inclination = np.deg2rad(INCLINATION_DEG)
    if raan is None:
        raan = np.deg2rad(RAAN_DEG)
    if argument_of_perigee is None:
        argument_of_perigee = np.deg2rad(ARGUMENT_OF_PERIGEE_DEG)

    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    radius = semi_latus_rectum / (
        1.0 + eccentricity * np.cos(true_anomaly)
    )
    argument_of_latitude = argument_of_perigee + true_anomaly

    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_inclination = np.cos(inclination)
    sin_inclination = np.sin(inclination)
    cos_u = np.cos(argument_of_latitude)
    sin_u = np.sin(argument_of_latitude)

    x = radius * (cos_raan * cos_u - sin_raan * sin_u * cos_inclination)
    y = radius * (sin_raan * cos_u + cos_raan * sin_u * cos_inclination)
    z = radius * sin_u * sin_inclination
    position = np.stack((x, y, z), axis=-1)

    velocity_scale = np.sqrt(EARTH_MU / semi_latus_rectum)
    velocity_perifocal_x = -velocity_scale * np.sin(true_anomaly)
    velocity_perifocal_y = velocity_scale * (
        eccentricity + np.cos(true_anomaly)
    )

    rotation_11 = cos_raan * np.cos(argument_of_perigee) - (
        sin_raan * np.sin(argument_of_perigee) * cos_inclination
    )
    rotation_12 = -cos_raan * np.sin(argument_of_perigee) - (
        sin_raan * np.cos(argument_of_perigee) * cos_inclination
    )
    rotation_21 = sin_raan * np.cos(argument_of_perigee) + (
        cos_raan * np.sin(argument_of_perigee) * cos_inclination
    )
    rotation_22 = -sin_raan * np.sin(argument_of_perigee) + (
        cos_raan * np.cos(argument_of_perigee) * cos_inclination
    )
    rotation_31 = np.sin(argument_of_perigee) * sin_inclination
    rotation_32 = np.cos(argument_of_perigee) * sin_inclination

    velocity_x = (
        rotation_11 * velocity_perifocal_x
        + rotation_12 * velocity_perifocal_y
    )
    velocity_y = (
        rotation_21 * velocity_perifocal_x
        + rotation_22 * velocity_perifocal_y
    )
    velocity_z = (
        rotation_31 * velocity_perifocal_x
        + rotation_32 * velocity_perifocal_y
    )
    velocity = np.stack((velocity_x, velocity_y, velocity_z), axis=-1)
    return position, velocity


def orbit_averaged_srp_element_rates(
    epoch: np.datetime64,
    semi_major_axes: np.ndarray,
    eccentricities: np.ndarray,
    raan: np.ndarray,
    argument_of_perigee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic orbit-averaged da/dt and de/dt from vector SRP.

    SRP is evaluated on the mean trajectory for each altitude and then shared
    by its Monte Carlo realizations. This avoids multiplying a deterministic
    optical force by the atmosphere ensemble while retaining a sufficiently
    fine orbit grid for eclipse entry and exit.
    """
    true_anomaly = solve_true_anomaly(
        SRP_MEAN_ANOMALY_SAMPLES[None, :],
        eccentricities[:, None],
    )
    position, velocity = orbital_state_eci(
        semi_major_axes[:, None],
        true_anomaly,
        raan[:, None],
        argument_of_perigee[:, None],
        eccentricities[:, None],
    )
    sun_position = sun_position_eci(epoch)
    spacecraft_to_sun = sun_position - position
    sun_distance = np.linalg.norm(spacecraft_to_sun, axis=-1)
    sunward_direction = spacecraft_to_sun / sun_distance[..., None]
    illumination = sunlight_fraction(position, sun_position)
    acceleration_magnitude = (
        SOLAR_RADIATION_PRESSURE_AT_1_AU
        * SOLAR_RADIATION_COEFFICIENT
        * SOLAR_RADIATION_AREA
        / SATELLITE_MASS
        * (ASTRONOMICAL_UNIT / sun_distance) ** 2
        * illumination
    )
    # Radiation pushes away from the Sun. It is not assigned the anti-velocity
    # direction used by a drag acceleration or by Low and Chia's scalar model.
    srp_acceleration = (
        -acceleration_magnitude[..., None] * sunward_direction
    )

    radius = np.linalg.norm(position, axis=-1)
    specific_power = np.sum(velocity * srp_acceleration, axis=-1)
    da_dt = (
        2.0
        * semi_major_axes[:, None] ** 2
        * specific_power
        / EARTH_MU
    )

    radial_direction = position / radius[..., None]
    angular_momentum_vector = np.cross(position, velocity)
    angular_momentum = np.linalg.norm(angular_momentum_vector, axis=-1)
    orbit_normal = angular_momentum_vector / angular_momentum[..., None]
    transverse_direction = np.cross(orbit_normal, radial_direction)
    radial_acceleration = np.sum(
        srp_acceleration * radial_direction, axis=-1
    )
    transverse_acceleration = np.sum(
        srp_acceleration * transverse_direction, axis=-1
    )
    eccentricity = eccentricities[:, None]
    semi_latus_rectum = semi_major_axes[:, None] * (
        1.0 - eccentricity**2
    )
    de_dt = (
        semi_latus_rectum
        * np.sin(true_anomaly)
        * radial_acceleration
        + (
            (semi_latus_rectum + radius) * np.cos(true_anomaly)
            + radius * eccentricity
        )
        * transverse_acceleration
    ) / angular_momentum

    # Both quadratures are uniform in mean anomaly and therefore in time.
    return np.mean(da_dt, axis=-1), np.mean(de_dt, axis=-1)


def j2_secular_rates(
    semi_major_axis: np.ndarray,
    eccentricity: np.ndarray | float = ECCENTRICITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return first-order J2 rates for RAAN and argument of perigee."""
    inclination = np.deg2rad(INCLINATION_DEG)
    semi_latus_rectum = semi_major_axis * (1.0 - eccentricity**2)
    mean_motion = np.sqrt(EARTH_MU / semi_major_axis**3)
    common = EARTH_J2 * mean_motion * (
        EARTH_EQUATORIAL_RADIUS / semi_latus_rectum
    ) ** 2
    raan_rate = -1.5 * common * np.cos(inclination)
    perigee_rate = 0.75 * common * (
        5.0 * np.cos(inclination) ** 2 - 1.0
    )
    return raan_rate, perigee_rate


def greenwich_mean_sidereal_angle(epoch: np.ndarray | np.datetime64) -> np.ndarray:
    """Return Greenwich mean sidereal angle in radians."""
    unix_seconds = (
        epoch - np.datetime64("1970-01-01T00:00:00", "s")
    ) / np.timedelta64(1, "s")
    julian_date = np.asarray(unix_seconds) / SECONDS_PER_DAY + 2_440_587.5
    centuries = (julian_date - 2_451_545.0) / 36_525.0
    gmst_degrees = (
        280.46061837
        + 360.98564736629 * (julian_date - 2_451_545.0)
        + 0.000387933 * centuries**2
        - centuries**3 / 38_710_000.0
    )
    return np.deg2rad(gmst_degrees % 360.0)


def eci_to_geodetic(
    position_eci: np.ndarray,
    epoch: np.ndarray | np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECI positions to WGS84 longitude, latitude, and altitude."""
    angle = greenwich_mean_sidereal_angle(epoch)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    x_eci = position_eci[..., 0]
    y_eci = position_eci[..., 1]
    z = position_eci[..., 2]
    x = cos_angle * x_eci + sin_angle * y_eci
    y = -sin_angle * x_eci + cos_angle * y_eci

    longitude = np.arctan2(y, x)
    distance_from_axis = np.hypot(x, y)
    bowring_angle = np.arctan2(
        z * EARTH_EQUATORIAL_RADIUS,
        distance_from_axis * EARTH_POLAR_RADIUS,
    )
    latitude = np.arctan2(
        z
        + EARTH_SECOND_ECCENTRICITY_SQUARED
        * EARTH_POLAR_RADIUS
        * np.sin(bowring_angle) ** 3,
        distance_from_axis
        - EARTH_ECCENTRICITY_SQUARED
        * EARTH_EQUATORIAL_RADIUS
        * np.cos(bowring_angle) ** 3,
    )
    prime_vertical_radius = EARTH_EQUATORIAL_RADIUS / np.sqrt(
        1.0 - EARTH_ECCENTRICITY_SQUARED * np.sin(latitude) ** 2
    )
    altitude = distance_from_axis / np.cos(latitude) - prime_vertical_radius

    return (
        np.rad2deg(longitude),
        np.rad2deg(latitude),
        altitude / 1000.0,
    )


def msis_atmosphere(
    epoch: np.ndarray | np.datetime64,
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude_km: np.ndarray,
    f107: np.ndarray | float,
    f107a: np.ndarray | float,
    ap: np.ndarray,
    version: float,
) -> np.ndarray:
    """Return all NRLMSIS variables in satellite fly-through mode."""
    original_shape = altitude_km.shape
    number_of_points = altitude_km.size
    dates = np.broadcast_to(epoch, original_shape).ravel()
    f107_values = np.broadcast_to(f107, original_shape).ravel()
    f107a_values = np.broadcast_to(f107a, original_shape).ravel()
    ap_values = np.broadcast_to(ap, original_shape + (7,)).reshape(
        number_of_points, 7
    )

    output = pymsis.calculate(
        dates,
        longitude.ravel(),
        latitude.ravel(),
        altitude_km.ravel(),
        f107s=f107_values,
        f107as=f107a_values,
        aps=ap_values,
        version=version,
        geomagnetic_activity=-1,
    ).reshape(original_shape + (len(Variable),))
    density = output[..., Variable.MASS_DENSITY]
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise RuntimeError("NRLMSIS returned a non-positive or invalid density.")
    return output


def msis_density(
    epoch: np.ndarray | np.datetime64,
    longitude: np.ndarray,
    latitude: np.ndarray,
    altitude_km: np.ndarray,
    f107: np.ndarray | float,
    f107a: np.ndarray | float,
    ap: np.ndarray,
    version: float,
) -> np.ndarray:
    """Evaluate and return total NRLMSIS mass density in kg/m^3."""
    return msis_atmosphere(
        epoch, longitude, latitude, altitude_km, f107, f107a, ap, version
    )[..., Variable.MASS_DENSITY]


def evaluate_density_and_sensitivities(
    epoch: np.datetime64,
    nominal_axes: np.ndarray,
    nominal_eccentricities: np.ndarray,
    raan: np.ndarray,
    argument_of_perigee: np.ndarray,
    need_f107_sensitivity: bool,
    need_f107a_sensitivity: bool,
    need_ap_sensitivity: bool,
) -> tuple[np.ndarray, ...]:
    """Evaluate NRLMSIS and local log-density gradients/curvatures."""
    true_anomaly = solve_true_anomaly(
        MEAN_ANOMALY_SAMPLES[None, :],
        nominal_eccentricities[:, None],
    )
    position, _ = orbital_state_eci(
        nominal_axes[:, None],
        true_anomaly,
        raan[:, None],
        argument_of_perigee[:, None],
        nominal_eccentricities[:, None],
    )
    orbital_period = 2.0 * pi * np.sqrt(nominal_axes**3 / EARTH_MU)
    sample_time_fraction = MEAN_ANOMALY_SAMPLES / (2.0 * pi) - 0.5
    sample_time_offsets = np.rint(
        orbital_period[:, None] * sample_time_fraction[None, :]
    ).astype("timedelta64[s]")
    sample_epochs = epoch + sample_time_offsets
    longitude, latitude, altitude_km = eci_to_geodetic(
        position, sample_epochs
    )
    # The storm-time Ap term changes every three hours. Each orbit point must
    # therefore use the drivers for its own UTC timestamp, including samples
    # lying either side of an interval boundary.
    sample_f107, sample_f107a, sample_ap = load_nominal_space_weather(
        sample_epochs.ravel()
    )
    sample_f107 = sample_f107.reshape(altitude_km.shape)
    sample_f107a = sample_f107a.reshape(altitude_km.shape)
    sample_ap = sample_ap.reshape(altitude_km.shape + (7,))

    # Pipeline the nominal atmosphere and required finite-difference cases
    # through one PyMSIS fly-through call. Weather sensitivities whose Monte
    # Carlo error is exactly zero are skipped; altitude sensitivity is always
    # needed as the realizations' trajectories diverge.
    case_names = ["nominal"]
    if need_f107_sensitivity:
        case_names.extend(("f107_minus", "f107_plus"))
    if need_f107a_sensitivity:
        case_names.extend(("f107a_minus", "f107a_plus"))
    if need_ap_sensitivity:
        case_names.append("ap")
    case_names.extend(("altitude_minus", "altitude_plus"))
    case_indices = {name: index for index, name in enumerate(case_names)}
    number_of_cases = len(case_names)
    case_shape = (number_of_cases,) + altitude_km.shape
    case_epochs = np.broadcast_to(sample_epochs, case_shape)
    case_longitude = np.broadcast_to(longitude, case_shape)
    case_latitude = np.broadcast_to(latitude, case_shape)
    case_altitude = np.broadcast_to(altitude_km, case_shape).copy()
    case_f107 = np.broadcast_to(sample_f107, case_shape).copy()
    case_f107a = np.broadcast_to(sample_f107a, case_shape).copy()
    case_ap = np.broadcast_to(
        sample_ap, case_shape + (7,)
    ).copy()

    if need_f107_sensitivity:
        case_f107[case_indices["f107_minus"]] -= F107_PERTURBATION
        case_f107[case_indices["f107_plus"]] += F107_PERTURBATION
    if need_f107a_sensitivity:
        case_f107a[case_indices["f107a_minus"]] -= F107A_PERTURBATION
        case_f107a[case_indices["f107a_plus"]] += F107A_PERTURBATION
    if need_ap_sensitivity:
        case_ap[case_indices["ap"]] = np.expm1(
            np.log1p(np.maximum(sample_ap, 0.0))
            + LOG_AP_PLUS_ONE_PERTURBATION
        )
    case_altitude[case_indices["altitude_minus"]] -= (
        ALTITUDE_PERTURBATION_KM
    )
    case_altitude[case_indices["altitude_plus"]] += (
        ALTITUDE_PERTURBATION_KM
    )

    atmosphere_cases = msis_atmosphere(
        case_epochs,
        case_longitude,
        case_latitude,
        case_altitude,
        case_f107,
        case_f107a,
        case_ap,
        MSIS_VERSION,
    )
    nominal_atmosphere = atmosphere_cases[0]
    reference_log_density = np.log(
        nominal_atmosphere[..., Variable.MASS_DENSITY]
    )
    zero_sensitivity = np.zeros_like(reference_log_density)

    def central_log_density_response(
        minus_case: str,
        plus_case: str,
        perturbation: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return centered first and second log-density derivatives."""
        log_minus = np.log(
            atmosphere_cases[
                case_indices[minus_case], ..., Variable.MASS_DENSITY
            ]
        )
        log_plus = np.log(
            atmosphere_cases[
                case_indices[plus_case], ..., Variable.MASS_DENSITY
            ]
        )
        gradient = (log_plus - log_minus) / (2.0 * perturbation)
        curvature = (
            log_plus - 2.0 * reference_log_density + log_minus
        ) / perturbation**2
        return gradient, curvature

    if need_f107_sensitivity:
        sensitivity_f107, curvature_f107 = central_log_density_response(
            "f107_minus", "f107_plus", F107_PERTURBATION
        )
    else:
        sensitivity_f107 = zero_sensitivity
        curvature_f107 = zero_sensitivity
    if need_f107a_sensitivity:
        sensitivity_f107a, curvature_f107a = central_log_density_response(
            "f107a_minus", "f107a_plus", F107A_PERTURBATION
        )
    else:
        sensitivity_f107a = zero_sensitivity
        curvature_f107a = zero_sensitivity
    if need_ap_sensitivity:
        density_ap = atmosphere_cases[
            case_indices["ap"], ..., Variable.MASS_DENSITY
        ]
        sensitivity_log_ap = (
            np.log(density_ap) - reference_log_density
        ) / LOG_AP_PLUS_ONE_PERTURBATION
    else:
        sensitivity_log_ap = zero_sensitivity
    sensitivity_altitude, curvature_altitude = central_log_density_response(
        "altitude_minus", "altitude_plus", ALTITUDE_PERTURBATION_KM
    )

    return (
        reference_log_density,
        sensitivity_f107,
        curvature_f107,
        sensitivity_f107a,
        curvature_f107a,
        sensitivity_log_ap,
        sensitivity_altitude,
        curvature_altitude,
        nominal_atmosphere,
        atmosphere_cases[case_indices["altitude_minus"]],
        atmosphere_cases[case_indices["altitude_plus"]],
    )


def initialize_space_weather_states(
    generator: np.random.Generator,
    correlation: np.ndarray,
) -> np.ndarray:
    """Draw stationary, jointly correlated standardized driver errors."""
    return generator.multivariate_normal(
        np.zeros(2),
        correlation,
        size=MONTE_CARLO_RUNS,
    )


def update_space_weather_states(
    states: np.ndarray,
    step: float,
    correlation: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Advance a multivariate Gauss-Markov process with unequal timescales."""
    persistence = np.exp(-step / SPACE_WEATHER_CORRELATION_TIMES)
    innovation_covariance = correlation * (
        1.0 - np.outer(persistence, persistence)
    )
    minimum_eigenvalue = np.min(np.linalg.eigvalsh(innovation_covariance))
    if minimum_eigenvalue < 0.0:
        innovation_covariance += np.eye(2) * (-minimum_eigenvalue + 1e-12)
    innovation_factor = np.linalg.cholesky(innovation_covariance)
    innovations = generator.standard_normal(states.shape) @ innovation_factor.T
    return states * persistence + innovations


def centered_running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centered running mean using reflected boundary conditions."""
    if window % 2 == 0:
        window += 1
    half_window = window // 2
    padded = np.pad(values, ((half_window, half_window), (0, 0)), mode="reflect")
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0))
    )
    return (cumulative[window:] - cumulative[:-window]) / window


def sample_space_weather_errors(
    steps: np.ndarray,
    correlation: np.ndarray,
    f107_observed_mask: np.ndarray,
    ap_observed_mask: np.ndarray,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample forecast errors while leaving observed drivers unperturbed."""
    if np.all(f107_observed_mask) and np.all(ap_observed_mask):
        shape = (len(steps), MONTE_CARLO_RUNS)
        zeros = np.zeros(shape)
        return zeros.copy(), zeros.copy(), zeros

    states = initialize_space_weather_states(generator, correlation)
    history = np.empty((len(steps), MONTE_CARLO_RUNS, 2))
    for index, step in enumerate(steps):
        states = update_space_weather_states(
            states, step, correlation, generator
        )
        history[index] = states

    f107_error = F107_SIGMA * history[:, :, 0]
    log_ap_error = LOG_AP_PLUS_ONE_SIGMA * history[:, :, 1]
    f107_error[f107_observed_mask] = 0.0
    log_ap_error[ap_observed_mask] = 0.0
    samples_per_81_days = int(
        round(81.0 * SECONDS_PER_DAY / PROPAGATION_STEP)
    )
    f107a_error = centered_running_mean(f107_error, samples_per_81_days)
    return f107_error, f107a_error, log_ap_error


def altitude_correlation_factor() -> np.ndarray:
    """Return Cholesky factor for residual errors across sampled altitudes."""
    separations = np.abs(
        SAMPLE_HEIGHTS_KM[:, None] - SAMPLE_HEIGHTS_KM[None, :]
    )
    correlation = np.exp(-separations / MODEL_ERROR_ALTITUDE_CORRELATION_KM)
    return np.linalg.cholesky(correlation + np.eye(len(SAMPLE_HEIGHTS_KM)) * 1e-12)


def update_model_error_states(
    states: np.ndarray,
    step: float,
    altitude_factor: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Advance temporally and spatially correlated unit-variance residuals."""
    persistence = np.exp(-step / MODEL_ERROR_TIME_CORRELATION)
    innovations = generator.standard_normal(states.shape) @ altitude_factor.T
    return persistence * states + np.sqrt(1.0 - persistence**2) * innovations


def model_log_sigma(altitude_km: np.ndarray, current_ap: float) -> np.ndarray:
    """Return assumed log-density standard deviation by height and activity."""
    coefficient_of_variation = (
        MODEL_ERROR_CV_AT_500_KM
        + MODEL_ERROR_CV_PER_100_KM * (altitude_km - 500.0) / 100.0
    )
    storm_multiplier = 1.0 + MODEL_ERROR_STORM_SCALE_PER_AP * max(
        current_ap - 15.0, 0.0
    )
    coefficient_of_variation = np.clip(
        coefficient_of_variation * storm_multiplier, 0.05, 0.60
    )
    return np.sqrt(np.log1p(coefficient_of_variation**2))


def sample_persistent_model_errors(
    generator: np.random.Generator,
    altitude_factor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw only persistent atmosphere/GSI model-discrepancy states."""
    density_bias_states = generator.standard_normal(
        (MONTE_CARLO_RUNS, len(SAMPLE_HEIGHTS_KM))
    ) @ altitude_factor.T
    drag_coefficient_bias_states = generator.standard_normal(MONTE_CARLO_RUNS)
    return density_bias_states, drag_coefficient_bias_states


def print_uncertainty_model() -> None:
    """Print the sampled inputs represented by the plotted one-sigma bars."""
    dimensions = " x ".join(f"{value:.3f}" for value in VELOX_BODY_DIMENSIONS)
    print(
        "Exact spacecraft inputs: "
        f"mass {SATELLITE_MASS:.3f} kg; dimensions {dimensions} m; "
        "broad face exactly normal to the relative flow."
    )
    print(
        f"Exact GSI inputs: wall {SURFACE_TEMPERATURE:.1f} K; Goodman "
        f"coefficient {GOODMAN_SUBSTRATE_COEFFICIENT:.3f}; surface molar "
        f"mass {1e3 * EFFECTIVE_SURFACE_MOLAR_MASS:.3f} g/mol."
    )
    print(
        "Exact vector-SRP inputs: "
        f"area {SOLAR_RADIATION_AREA:.3f} m^2; coefficient "
        f"{SOLAR_RADIATION_COEFFICIENT:.3f}; finite-disc Earth eclipses."
    )
    print("Environmental/model uncertainty propagated per trajectory:")
    print(
        f"  persistent density/GSI-model scale CV: "
        f"{DENSITY_SYSTEMATIC_CV:.0%}/{DRAG_COEFFICIENT_SYSTEMATIC_CV:.0%}; "
        f"residual density CV at "
        f"500 km: {MODEL_ERROR_CV_AT_500_KM:.0%}"
    )
    print(
        f"  horizontal wind sigma: {NEUTRAL_WIND_SIGMA_M_PER_S:.0f} m/s"
    )
    print(
        "  forecast-only F10.7/log(Ap+1) errors: sigma "
        f"{F107_SIGMA:.1f}/{LOG_AP_PLUS_ONE_SIGMA:.2f}; observed drivers "
        "are not artificially perturbed."
    )


def initialize_neutral_wind_states(
    generator: np.random.Generator,
    altitude_factor: np.ndarray,
) -> np.ndarray:
    """Draw correlated standardized east/north neutral-wind errors."""
    independent = generator.standard_normal(
        (MONTE_CARLO_RUNS, len(SAMPLE_HEIGHTS_KM), 2)
    )
    return np.einsum("rhq,kh->rkq", independent, altitude_factor)


def update_neutral_wind_states(
    states: np.ndarray,
    step: float,
    altitude_factor: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Advance unresolved horizontal wind with a Gauss-Markov process."""
    persistence = np.exp(-step / NEUTRAL_WIND_TIME_CORRELATION)
    independent = generator.standard_normal(states.shape)
    innovations = np.einsum("rhq,kh->rkq", independent, altitude_factor)
    return (
        persistence * states
        + np.sqrt(1.0 - persistence**2) * innovations
    )


def assemble_density_realizations(
    reference_log_density: np.ndarray,
    sensitivity_f107: np.ndarray,
    curvature_f107: np.ndarray,
    sensitivity_f107a: np.ndarray,
    curvature_f107a: np.ndarray,
    sensitivity_log_ap: np.ndarray,
    sensitivity_altitude: np.ndarray,
    curvature_altitude: np.ndarray,
    semi_major_axes: np.ndarray,
    nominal_axes: np.ndarray,
    eccentricities: np.ndarray,
    nominal_eccentricities: np.ndarray,
    f107_error: np.ndarray,
    f107a_error: np.ndarray,
    log_ap_error: np.ndarray,
    model_error_states: np.ndarray,
    density_bias_states: np.ndarray,
    current_ap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble positive density samples from physical and model errors."""
    delta_f107 = f107_error[:, None, None]
    delta_f107a = f107a_error[:, None, None]
    delta_log_ap = log_ap_error[:, None, None]

    realization_true_anomaly = solve_true_anomaly(
        MEAN_ANOMALY_SAMPLES[None, None, :],
        eccentricities[:, :, None],
    )
    realization_radius = (
        semi_major_axes[:, :, None] * (1.0 - eccentricities[:, :, None] ** 2)
        / (
            1.0
            + eccentricities[:, :, None]
            * np.cos(realization_true_anomaly)
        )
    )
    nominal_true_anomaly = solve_true_anomaly(
        MEAN_ANOMALY_SAMPLES[None, :],
        nominal_eccentricities[:, None],
    )
    nominal_radius = (
        nominal_axes[:, None] * (1.0 - nominal_eccentricities[:, None] ** 2)
        / (
            1.0
            + nominal_eccentricities[:, None]
            * np.cos(nominal_true_anomaly)
        )
    )
    altitude_difference_km = (
        realization_radius - nominal_radius[None, :, :]
    ) / 1000.0

    nominal_altitudes = (nominal_axes - EARTH_EQUATORIAL_RADIUS) / 1000.0
    sigma_log_density = model_log_sigma(nominal_altitudes, current_ap)
    residual = (
        sigma_log_density[None, :, None] * model_error_states[:, :, None]
        - 0.5 * sigma_log_density[None, :, None] ** 2
    )
    systematic_log_sigma = np.sqrt(np.log1p(DENSITY_SYSTEMATIC_CV**2))
    systematic_bias = (
        systematic_log_sigma * density_bias_states[:, :, None]
        - 0.5 * systematic_log_sigma**2
    )

    log_density = np.broadcast_to(
        reference_log_density,
        (len(f107_error),) + reference_log_density.shape,
    ).copy()
    log_density += sensitivity_f107[None, :, :] * delta_f107
    log_density += 0.5 * curvature_f107[None, :, :] * delta_f107**2
    log_density += sensitivity_f107a[None, :, :] * delta_f107a
    log_density += 0.5 * curvature_f107a[None, :, :] * delta_f107a**2
    log_density += sensitivity_log_ap[None, :, :] * delta_log_ap
    log_density += (
        sensitivity_altitude[None, :, :] * altitude_difference_km
    )
    log_density += (
        0.5
        * curvature_altitude[None, :, :]
        * altitude_difference_km**2
    )
    log_density += residual
    log_density += systematic_bias
    return (
        np.exp(log_density),
        realization_true_anomaly,
        altitude_difference_km,
    )


def error_function_nonnegative(values: np.ndarray) -> np.ndarray:
    """Return erf(x) for nonnegative speed ratios without wasted tail work."""
    if np.any(values < 0.0):
        raise ValueError("Speed ratios supplied to erf must be nonnegative.")
    result = np.ones_like(values)
    # For x >= 6, erfc(x) < 2.2e-17 and erf(x) rounds to exactly one in
    # float64. Evaluate Abramowitz-Stegun 7.1.26 only on the material subset.
    material = values < 6.0
    selected = values[material]
    t = 1.0 / (1.0 + 0.3275911 * selected)
    erfc_selected = (
        (
            (
                (
                    (1.061405429 * t - 1.453152027) * t
                    + 1.421413741
                )
                * t
                - 0.284496736
            )
            * t
            + 0.254829592
        )
        * t
        * np.exp(-(selected**2))
    )
    result[material] = 1.0 - erfc_selected
    return result


def sentman_drag_coefficient(
    relative_speed: np.ndarray,
    atmosphere: np.ndarray,
) -> np.ndarray:
    """Return exact-parameter VELOX free-molecular drag coefficients.

    The known broad face is normal to the relative flow. Opposite box faces are
    summed analytically with Sentman's diffuse-reflection, incomplete-
    accommodation solution. Atomic-oxygen coverage follows a Langmuir isotherm;
    clean-surface accommodation follows Goodman. Each NRLMSIS species is
    evaluated separately and mass weighted. Atmosphere may omit the leading
    Monte Carlo axis because its nominal composition is common to realizations.
    """
    temperature = np.maximum(
        atmosphere[..., Variable.TEMPERATURE], 1.0
    )
    number_density = np.nan_to_num(
        atmosphere[..., list(MSIS_DRAG_SPECIES)],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    speed_ratio = relative_speed[..., None] / np.sqrt(
        2.0
        * UNIVERSAL_GAS_CONSTANT
        * temperature[..., None]
        / SPECIES_MOLAR_MASS
    )
    speed_ratio = np.maximum(speed_ratio, 1e-6)

    atomic_oxygen_density = np.nan_to_num(
        atmosphere[..., Variable.O], nan=0.0, posinf=0.0, neginf=0.0
    )
    atomic_oxygen_pressure = (
        atomic_oxygen_density * BOLTZMANN_CONSTANT * temperature
    )
    adsorption_product = LANGMUIR_DRIA_CONSTANT * atomic_oxygen_pressure
    surface_coverage = adsorption_product / (1.0 + adsorption_product)

    mass_ratio = SPECIES_MOLAR_MASS / EFFECTIVE_SURFACE_MOLAR_MASS
    clean_accommodation = (
        GOODMAN_SUBSTRATE_COEFFICIENT
        * mass_ratio
        / (1.0 + mass_ratio) ** 2
    )
    clean_accommodation = np.clip(clean_accommodation, 0.0, 1.0)
    incident_temperature = (
        SPECIES_MOLAR_MASS
        * relative_speed[..., None] ** 2
        / (3.0 * UNIVERSAL_GAS_CONSTANT)
    )
    clean_reflected_temperature = (
        incident_temperature * (1.0 - clean_accommodation)
        + SURFACE_TEMPERATURE * clean_accommodation
    )
    length_x, length_y, length_z = VELOX_BODY_DIMENSIONS
    broad_face_ratio = length_y * length_z / CROSS_SECTIONAL_AREA
    side_face_ratio = (
        length_x * length_z + length_x * length_y
    ) / CROSS_SECTIONAL_AREA

    # For exact ram pointing, the broad-face direction cosine is one and both
    # side-pair cosines are zero. Applying these values before array expansion
    # avoids allocating an unnecessary three-axis tensor.
    error_function = error_function_nonnegative(speed_ratio)
    base_coefficient = broad_face_ratio * (
        2.0
        * error_function
        * (1.0 + 1.0 / (2.0 * speed_ratio**2))
        + 2.0
        * np.exp(-(speed_ratio**2))
        / (speed_ratio * np.sqrt(pi))
    )
    base_coefficient += (
        side_face_ratio * 2.0 / (speed_ratio * np.sqrt(pi))
    )
    reflection_prefactor = (
        broad_face_ratio * np.sqrt(pi) / speed_ratio
    )
    clean_coefficient = base_coefficient + reflection_prefactor * np.sqrt(
        clean_reflected_temperature / temperature[..., None]
    )
    adsorbed_coefficient = base_coefficient + reflection_prefactor * np.sqrt(
        SURFACE_TEMPERATURE / temperature[..., None]
    )
    effective_species_coefficient = (
        (1.0 - surface_coverage[..., None]) * clean_coefficient
        + surface_coverage[..., None] * adsorbed_coefficient
    )
    species_mass_weights = number_density * SPECIES_MOLAR_MASS
    total_mass_weight = np.sum(species_mass_weights, axis=-1)
    if np.any(total_mass_weight <= 0.0):
        raise RuntimeError("NRLMSIS returned no usable neutral composition.")

    physical_coefficient = np.sum(
        effective_species_coefficient * species_mass_weights,
        axis=-1,
    ) / total_mass_weight
    if np.any(~np.isfinite(physical_coefficient)) or np.any(
        physical_coefficient <= 0.0
    ):
        raise RuntimeError("The physical drag-coefficient model became invalid.")
    return physical_coefficient


def orbit_averaged_element_rates(
    semi_major_axes: np.ndarray,
    eccentricities: np.ndarray,
    true_anomaly: np.ndarray,
    densities: np.ndarray,
    altitude_differences_km: np.ndarray,
    atmosphere: np.ndarray,
    atmosphere_altitude_minus: np.ndarray,
    atmosphere_altitude_plus: np.ndarray,
    drag_coefficient_bias_states: np.ndarray,
    neutral_wind_states: np.ndarray,
    raan: np.ndarray,
    argument_of_perigee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return orbit-averaged da/dt and de/dt from aerodynamic drag."""
    position, velocity = orbital_state_eci(
        semi_major_axes[:, :, None],
        true_anomaly,
        raan[None, :, None],
        argument_of_perigee[None, :, None],
        eccentricities[:, :, None],
    )
    atmosphere_velocity = np.empty_like(position)
    atmosphere_velocity[..., 0] = -EARTH_ROTATION_RATE * position[..., 1]
    atmosphere_velocity[..., 1] = EARTH_ROTATION_RATE * position[..., 0]
    atmosphere_velocity[..., 2] = 0.0

    # Add realization-specific unresolved horizontal winds in the local east
    # and north directions. These states are correlated in time and altitude.
    radius = np.linalg.norm(position, axis=-1)
    distance_from_axis = np.maximum(
        np.hypot(position[..., 0], position[..., 1]), 1.0
    )
    east_direction = np.empty_like(position)
    east_direction[..., 0] = -position[..., 1] / distance_from_axis
    east_direction[..., 1] = position[..., 0] / distance_from_axis
    east_direction[..., 2] = 0.0
    north_direction = np.empty_like(position)
    north_direction[..., 0] = (
        -position[..., 0]
        * position[..., 2]
        / (radius * distance_from_axis)
    )
    north_direction[..., 1] = (
        -position[..., 1]
        * position[..., 2]
        / (radius * distance_from_axis)
    )
    north_direction[..., 2] = distance_from_axis / radius
    zonal_wind = (
        NEUTRAL_WIND_SIGMA_M_PER_S
        * neutral_wind_states[..., 0, None, None]
    )
    meridional_wind = (
        NEUTRAL_WIND_SIGMA_M_PER_S
        * neutral_wind_states[..., 1, None, None]
    )
    atmosphere_velocity += (
        zonal_wind * east_direction + meridional_wind * north_direction
    )

    relative_velocity = velocity - atmosphere_velocity
    relative_speed = np.linalg.norm(relative_velocity, axis=-1)
    drag_coefficient = sentman_drag_coefficient(
        relative_speed,
        atmosphere,
    )

    # Correct Cd for realization-specific altitude through NRLMSIS composition
    # and temperature, not density alone. The atmospheric derivatives are
    # evaluated at the orbit-sample mean speed without a Monte Carlo axis; wind
    # changes speed by only about 0.5%, while Cd still uses every realization's
    # exact speed above. This retains the important composition response at
    # roughly 1/200 of the cost of two full-ensemble Sentman evaluations.
    reference_speed = np.mean(relative_speed, axis=0)
    reference_drag_coefficient = sentman_drag_coefficient(
        reference_speed, atmosphere
    )
    drag_coefficient_altitude_minus = sentman_drag_coefficient(
        reference_speed, atmosphere_altitude_minus
    )
    drag_coefficient_altitude_plus = sentman_drag_coefficient(
        reference_speed, atmosphere_altitude_plus
    )
    log_reference_drag_coefficient = np.log(reference_drag_coefficient)
    log_drag_coefficient_minus = np.log(
        drag_coefficient_altitude_minus
    )
    log_drag_coefficient_plus = np.log(
        drag_coefficient_altitude_plus
    )
    drag_altitude_gradient = (
        log_drag_coefficient_plus - log_drag_coefficient_minus
    ) / (2.0 * ALTITUDE_PERTURBATION_KM)
    drag_altitude_curvature = (
        log_drag_coefficient_plus
        - 2.0 * log_reference_drag_coefficient
        + log_drag_coefficient_minus
    ) / ALTITUDE_PERTURBATION_KM**2
    drag_coefficient *= np.exp(
        drag_altitude_gradient[None, :, :] * altitude_differences_km
        + 0.5
        * drag_altitude_curvature[None, :, :]
        * altitude_differences_km**2
    )
    drag_log_sigma = np.sqrt(
        np.log1p(DRAG_COEFFICIENT_SYSTEMATIC_CV**2)
    )
    drag_coefficient *= np.exp(
        drag_log_sigma
        * drag_coefficient_bias_states[:, None, None]
        - 0.5 * drag_log_sigma**2
    )

    drag_scale = (
        0.5
        * drag_coefficient
        * densities
        * CROSS_SECTIONAL_AREA
        * relative_speed
        / SATELLITE_MASS
    )
    drag_acceleration = -drag_scale[..., None] * relative_velocity
    specific_power = np.sum(velocity * drag_acceleration, axis=-1)
    da_dt = 2.0 * semi_major_axes[:, :, None] ** 2 * specific_power / EARTH_MU

    radial_direction = position / radius[..., None]
    angular_momentum_vector = np.cross(position, velocity)
    angular_momentum = np.linalg.norm(angular_momentum_vector, axis=-1)
    orbit_normal = angular_momentum_vector / angular_momentum[..., None]
    transverse_direction = np.cross(orbit_normal, radial_direction)
    radial_acceleration = np.sum(
        drag_acceleration * radial_direction, axis=-1
    )
    transverse_acceleration = np.sum(
        drag_acceleration * transverse_direction, axis=-1
    )
    eccentricity = eccentricities[:, :, None]
    semi_latus_rectum = (
        semi_major_axes[:, :, None] * (1.0 - eccentricity**2)
    )
    de_dt = (
        semi_latus_rectum
        * np.sin(true_anomaly)
        * radial_acceleration
        + (
            (semi_latus_rectum + radius) * np.cos(true_anomaly)
            + radius * eccentricity
        )
        * transverse_acceleration
    ) / angular_momentum

    # Samples are equally spaced in mean anomaly, hence equally spaced in time.
    return np.mean(da_dt, axis=-1), np.mean(de_dt, axis=-1)


def run_monte_carlo(
    launch_epoch: np.datetime64,
    use_cache: bool = ENABLE_RESULT_CACHE,
    simulation_duration: float = SIMULATION_DURATION,
    verbose: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Return daily-equivalent decay rates for all realizations and heights."""
    if use_cache:
        cached_decay_rates = load_cached_decay_rates(
            launch_epoch, simulation_duration
        )
        if cached_decay_rates is not None:
            if progress_callback is not None:
                progress_callback(1, 1)
            return cached_decay_rates

    dates, steps = build_time_grid(launch_epoch, simulation_duration)
    if progress_callback is not None:
        progress_callback(0, len(steps))
    _, _, ap_values = load_nominal_space_weather(dates)
    f107_observed_mask, ap_observed_mask = observed_space_weather_masks(dates)
    uncertain_driver_mask = ~(f107_observed_mask & ap_observed_mask)
    if not np.any(uncertain_driver_mask):
        correlation = np.eye(2)
        if verbose:
            print(
                "All space-weather inputs are observed; no forecast-error "
                "perturbation is applied."
            )
    else:
        correlation, correlation_lower, correlation_upper, sample_size = (
            estimate_driver_innovation_correlation(launch_epoch)
        )
        if verbose:
            print_driver_correlation_estimate(
                correlation,
                correlation_lower,
                correlation_upper,
                sample_size,
            )
            print(
                f"Forecast-error perturbations apply to "
                f"{np.count_nonzero(uncertain_driver_mask)}/"
                f"{len(uncertain_driver_mask)} steps."
            )
    if verbose:
        print(
            "Published VELOX-C1 observed/reference validation ratio "
            "(reported, not applied): "
            f"{PUBLISHED_VALIDATION_RATIO:.5f} "
            f"({VELOX_OBSERVED_DECAY_M / 1000.0:.3f}/"
            f"{VELOX_REFERENCE_MODEL_DECAY_M / 1000.0:.3f} km)"
        )
        print_uncertainty_model()

    generator = np.random.default_rng(RANDOM_SEED)
    altitude_factor = altitude_correlation_factor()
    density_bias_states, drag_coefficient_bias_states = (
        sample_persistent_model_errors(generator, altitude_factor)
    )
    initial_axes = EARTH_EQUATORIAL_RADIUS + SAMPLE_HEIGHTS_KM * 1000.0
    semi_major_axes = np.broadcast_to(
        initial_axes, (MONTE_CARLO_RUNS, len(SAMPLE_HEIGHTS_KM))
    ).copy()
    eccentricities = np.broadcast_to(
        ECCENTRICITY,
        semi_major_axes.shape,
    ).copy()
    raan = np.full(len(SAMPLE_HEIGHTS_KM), np.deg2rad(RAAN_DEG))
    argument_of_perigee = np.full(
        len(SAMPLE_HEIGHTS_KM), np.deg2rad(ARGUMENT_OF_PERIGEE_DEG)
    )

    f107_errors, f107a_errors, log_ap_errors = sample_space_weather_errors(
        steps,
        correlation,
        f107_observed_mask,
        ap_observed_mask,
        generator,
    )
    model_error_states = generator.standard_normal(
        (MONTE_CARLO_RUNS, len(SAMPLE_HEIGHTS_KM))
    ) @ altitude_factor.T
    neutral_wind_states = initialize_neutral_wind_states(
        generator, altitude_factor
    )
    reentry_radius = EARTH_EQUATORIAL_RADIUS + REENTRY_ALTITUDE
    reentered = np.zeros_like(semi_major_axes, dtype=bool)

    for step_index, (epoch, step) in enumerate(zip(dates, steps)):
        model_error_states = update_model_error_states(
            model_error_states, step, altitude_factor, generator
        )
        neutral_wind_states = update_neutral_wind_states(
            neutral_wind_states,
            step,
            altitude_factor,
            generator,
        )

        nominal_axes = np.mean(semi_major_axes, axis=0)
        nominal_eccentricities = np.mean(eccentricities, axis=0)
        start_raan_rate, start_perigee_rate = j2_secular_rates(
            nominal_axes, nominal_eccentricities
        )
        midpoint_raan = np.mod(
            raan + 0.5 * start_raan_rate * step, 2.0 * pi
        )
        midpoint_argument_of_perigee = np.mod(
            argument_of_perigee + 0.5 * start_perigee_rate * step,
            2.0 * pi,
        )
        need_f107_sensitivity = np.any(f107_errors[step_index] != 0.0)
        need_f107a_sensitivity = np.any(f107a_errors[step_index] != 0.0)
        need_ap_sensitivity = np.any(log_ap_errors[step_index] != 0.0)
        (
            reference_log_density,
            sensitivity_f107,
            curvature_f107,
            sensitivity_f107a,
            curvature_f107a,
            sensitivity_log_ap,
            sensitivity_altitude,
            curvature_altitude,
            nominal_atmosphere,
            atmosphere_altitude_minus,
            atmosphere_altitude_plus,
        ) = evaluate_density_and_sensitivities(
            epoch,
            nominal_axes,
            nominal_eccentricities,
            midpoint_raan,
            midpoint_argument_of_perigee,
            need_f107_sensitivity,
            need_f107a_sensitivity,
            need_ap_sensitivity,
        )

        (
            densities,
            realization_true_anomaly,
            altitude_differences_km,
        ) = assemble_density_realizations(
            reference_log_density,
            sensitivity_f107,
            curvature_f107,
            sensitivity_f107a,
            curvature_f107a,
            sensitivity_log_ap,
            sensitivity_altitude,
            curvature_altitude,
            semi_major_axes,
            nominal_axes,
            eccentricities,
            nominal_eccentricities,
            f107_errors[step_index],
            f107a_errors[step_index],
            log_ap_errors[step_index],
            model_error_states,
            density_bias_states,
            float(ap_values[step_index, 1]),
        )
        drag_da_dt, drag_de_dt = orbit_averaged_element_rates(
            semi_major_axes,
            eccentricities,
            realization_true_anomaly,
            densities,
            altitude_differences_km,
            nominal_atmosphere,
            atmosphere_altitude_minus,
            atmosphere_altitude_plus,
            drag_coefficient_bias_states,
            neutral_wind_states,
            midpoint_raan,
            midpoint_argument_of_perigee,
        )
        midpoint_epoch = epoch + np.timedelta64(
            int(round(0.5 * step)), "s"
        )
        srp_da_dt, srp_de_dt = orbit_averaged_srp_element_rates(
            midpoint_epoch,
            nominal_axes,
            nominal_eccentricities,
            midpoint_raan,
            midpoint_argument_of_perigee,
        )
        # SRP is a deterministic vector perturbation for the exact spacecraft
        # parameters. Its mean-orbit rates are common to the environmental
        # Monte Carlo realizations; drag retains realization-specific physics.
        da_dt = drag_da_dt + srp_da_dt[None, :]
        de_dt = drag_de_dt + srp_de_dt[None, :]
        semi_major_axes += np.where(reentered, 0.0, da_dt * step)
        eccentricities += np.where(reentered, 0.0, de_dt * step)
        eccentricities = np.clip(eccentricities, 0.0, 0.99)
        newly_reentered = (
            semi_major_axes * (1.0 - eccentricities) <= reentry_radius
        )
        minimum_axis = reentry_radius / (1.0 - eccentricities)
        semi_major_axes = np.maximum(semi_major_axes, minimum_axis)
        reentered |= newly_reentered

        # Advance the mean orbit orientation. J2 does not secularly change
        # inclination at first order, but its node/perigee precession changes
        # the longitude and local-solar-time regions sampled by NRLMSIS.
        representative_axes = np.mean(semi_major_axes, axis=0)
        representative_eccentricities = np.mean(eccentricities, axis=0)
        end_raan_rate, end_perigee_rate = j2_secular_rates(
            representative_axes, representative_eccentricities
        )
        raan = np.mod(
            raan + 0.5 * (start_raan_rate + end_raan_rate) * step,
            2.0 * pi,
        )
        argument_of_perigee = np.mod(
            argument_of_perigee
            + 0.5 * (start_perigee_rate + end_perigee_rate) * step,
            2.0 * pi,
        )

        if verbose and (
            (step_index + 1) % 250 == 0 or step_index + 1 == len(steps)
        ):
            print(
                f"Completed {step_index + 1}/{len(steps)} atmospheric steps",
                end="\r",
            )
        if progress_callback is not None:
            progress_callback(step_index + 1, len(steps))

    if verbose:
        print()
    total_decay_m = initial_axes[None, :] - semi_major_axes
    decay_rates = total_decay_m / simulation_duration * SECONDS_PER_DAY
    if use_cache:
        save_cached_decay_rates(
            launch_epoch, decay_rates, simulation_duration
        )
    return decay_rates


def summarize_decay_rates(decay_rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Monte Carlo means and one-sample-standard-deviation errors."""
    mean = np.mean(decay_rates, axis=0)
    standard_deviation = np.std(decay_rates, axis=0, ddof=1)
    return mean, standard_deviation


def smooth_trend_curve(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    sample_count: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a shape-preserving cubic curve through ordered samples.

    The Fritsch-Carlson derivative construction preserves the direction and
    local shape of monotonic data. It therefore smooths the displayed line
    without adding oscillations or changing the underlying simulation values.
    """
    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("Trend samples must be equally sized one-dimensional arrays.")
    if x.size < 2:
        raise ValueError("At least two samples are required for a trend curve.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Trend samples must contain only finite values.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    interval_widths = np.diff(x)
    if np.any(interval_widths <= 0.0):
        raise ValueError("Trend sample coordinates must be unique.")

    secant_slopes = np.diff(y) / interval_widths
    derivatives = np.zeros_like(y)
    if x.size == 2:
        derivatives[:] = secant_slopes[0]
    else:
        matching_direction = secant_slopes[:-1] * secant_slopes[1:] > 0.0
        left_weights = 2.0 * interval_widths[1:] + interval_widths[:-1]
        right_weights = interval_widths[1:] + 2.0 * interval_widths[:-1]
        derivatives[1:-1][matching_direction] = (
            left_weights[matching_direction] + right_weights[matching_direction]
        ) / (
            left_weights[matching_direction] / secant_slopes[:-1][matching_direction]
            + right_weights[matching_direction] / secant_slopes[1:][matching_direction]
        )

        def endpoint_derivative(
            first_width: float,
            second_width: float,
            first_slope: float,
            second_slope: float,
        ) -> float:
            """Limit one endpoint derivative to preserve the adjacent shape."""
            derivative = (
                (2.0 * first_width + second_width) * first_slope
                - first_width * second_slope
            ) / (first_width + second_width)
            if np.sign(derivative) != np.sign(first_slope):
                return 0.0
            if (
                np.sign(first_slope) != np.sign(second_slope)
                and abs(derivative) > abs(3.0 * first_slope)
            ):
                return 3.0 * first_slope
            return float(derivative)

        derivatives[0] = endpoint_derivative(
            interval_widths[0],
            interval_widths[1],
            secant_slopes[0],
            secant_slopes[1],
        )
        derivatives[-1] = endpoint_derivative(
            interval_widths[-1],
            interval_widths[-2],
            secant_slopes[-1],
            secant_slopes[-2],
        )

    smooth_x = np.linspace(x[0], x[-1], max(int(sample_count), x.size))
    interval_indices = np.searchsorted(x, smooth_x, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, x.size - 2)
    widths = interval_widths[interval_indices]
    fractions = (smooth_x - x[interval_indices]) / widths
    fractions_squared = fractions * fractions
    fractions_cubed = fractions_squared * fractions

    smooth_y = (
        (2.0 * fractions_cubed - 3.0 * fractions_squared + 1.0)
        * y[interval_indices]
        + (fractions_cubed - 2.0 * fractions_squared + fractions)
        * widths
        * derivatives[interval_indices]
        + (-2.0 * fractions_cubed + 3.0 * fractions_squared)
        * y[interval_indices + 1]
        + (fractions_cubed - fractions_squared)
        * widths
        * derivatives[interval_indices + 1]
    )
    return smooth_x, smooth_y


def exponential_regression_curve(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    standard_deviation: np.ndarray | None = None,
    sample_count: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and evaluate a weighted exponential altitude regression.

    Orbital decay driven by thermospheric drag is approximately exponential
    with altitude. The regression is therefore linear in log decay rate.
    Supplied Monte Carlo standard deviations are converted to approximate
    log-space uncertainties and used as inverse-uncertainty weights.
    """
    x = np.asarray(sample_x, dtype=float)
    y = np.asarray(sample_y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(
            "Regression samples must be equally sized one-dimensional arrays."
        )
    if x.size < 2:
        raise ValueError("At least two samples are required for regression.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Regression samples must contain only finite values.")
    if np.any(y <= 0.0):
        raise ValueError("Exponential regression requires positive rates.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("Regression sample coordinates must be unique.")

    altitude_reference = float(np.mean(x))
    altitude_scale = float(np.ptp(x))
    normalized_x = (x - altitude_reference) / altitude_scale
    design_matrix = np.column_stack((np.ones_like(normalized_x), normalized_x))
    log_rates = np.log(y)
    weights = np.ones_like(y)

    if standard_deviation is not None:
        sigma = np.asarray(standard_deviation, dtype=float)
        if sigma.shape != np.asarray(sample_y).shape:
            raise ValueError("Regression uncertainties must match the rate samples.")
        sigma = sigma[order]
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("Regression uncertainties must be finite and nonnegative.")
        if np.all(sigma > 0.0):
            log_sigma = np.sqrt(np.log1p((sigma / y) ** 2))
            weights = 1.0 / log_sigma

    weighted_design = design_matrix * weights[:, None]
    weighted_rates = log_rates * weights
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design,
        weighted_rates,
        rcond=None,
    )

    regression_x = np.linspace(x[0], x[-1], max(int(sample_count), x.size))
    normalized_regression_x = (
        regression_x - altitude_reference
    ) / altitude_scale
    regression_y = np.exp(
        coefficients[0] + coefficients[1] * normalized_regression_x
    )
    return regression_x, regression_y


def regression_trend_with_uncertainty(
    heights_km: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    sample_count: int = 401,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a fitted mean and display-only one-sigma uncertainty envelope."""
    regression_heights, regression_mean = exponential_regression_curve(
        heights_km,
        mean,
        standard_deviation,
        sample_count,
    )
    relative_sigma = np.asarray(standard_deviation, dtype=float) / np.asarray(
        mean,
        dtype=float,
    )
    _, smooth_relative_sigma = smooth_trend_curve(
        heights_km,
        relative_sigma,
        sample_count,
    )
    smooth_relative_sigma = np.maximum(0.0, smooth_relative_sigma)
    lower_bound = np.maximum(
        0.0,
        regression_mean * (1.0 - smooth_relative_sigma),
    )
    upper_bound = regression_mean * (1.0 + smooth_relative_sigma)
    return regression_heights, regression_mean, lower_bound, upper_bound


def print_summary(
    heights_km: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> None:
    """Print the values represented by the graph and uncertainty band."""
    print("\nAltitude   Mean decay rate   Standard deviation")
    print("  (km)         (m/day)             (m/day)")
    for height, centre, uncertainty in zip(
        heights_km, mean, standard_deviation
    ):
        print(f"{height:6.0f} {centre:17.3f} {uncertainty:20.3f}")


def plot_decay_rates(
    heights_km: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    launch_epoch: np.datetime64,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot sampled means with a weighted regression and one-sigma band."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 6))
    smooth_heights, smooth_mean, lower_bound, upper_bound = (
        regression_trend_with_uncertainty(
            heights_km,
            mean,
            standard_deviation,
        )
    )
    axis.fill_between(
        smooth_heights,
        lower_bound,
        upper_bound,
        color="tab:red",
        alpha=0.14,
        linewidth=0.0,
    )
    axis.plot(
        smooth_heights,
        smooth_mean,
        color="tab:red",
        linewidth=2.2,
        label="Weighted exponential regression +/- 1 standard deviation",
    )
    axis.scatter(
        heights_km,
        mean,
        color="tab:red",
        marker="o",
        s=24,
        zorder=3,
    )
    launch_day = np.datetime_as_string(launch_epoch, unit="D")
    axis.set_title(
        "VELOX-C1 NRLMSIS/Sentman + vector-SRP decay rate "
        f"({MONTE_CARLO_RUNS} realizations)\n"
        f"One-year propagation from {launch_day} UTC"
    )
    axis.set_xlabel("Initial altitude (km)")
    axis.set_ylabel("Mean semi-major-axis decay rate (m/day)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is None:
        output_path = (
            RUNTIME_DIRECTORY / "orbital_decay_rate_with_uncertainty.png"
        )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Graph saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_solar_cycle_comparison(
    results: list[tuple[str, np.datetime64, np.ndarray, np.ndarray, str]],
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Overlay instantaneous solar-cycle decay curves with 1-sigma bands."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 7))
    for phase, epoch, mean, standard_deviation, colour in results:
        epoch_day = np.datetime_as_string(epoch, unit="D")
        smooth_heights, smooth_mean, lower_bound, upper_bound = (
            regression_trend_with_uncertainty(
                SAMPLE_HEIGHTS_KM,
                mean,
                standard_deviation,
            )
        )
        axis.plot(
            smooth_heights,
            smooth_mean,
            color=colour,
            linewidth=2.2,
            label=f"{phase}: {epoch_day} (mean +/- 1 sigma)",
        )
        axis.scatter(
            SAMPLE_HEIGHTS_KM,
            mean,
            color=colour,
            marker="o",
            s=24,
            zorder=3,
        )
        axis.fill_between(
            smooth_heights,
            lower_bound,
            upper_bound,
            color=colour,
            alpha=0.14,
            linewidth=0.0,
        )

    axis.set_title(
        "VELOX-C1 orbital decay rate across Solar Cycle 25\n"
        f"Instantaneous orbit-averaged NRLMSIS/Sentman + vector-SRP, "
        f"{MONTE_CARLO_RUNS} "
        "realizations per epoch"
    )
    axis.set_xlabel("Initial altitude (km)")
    axis.set_ylabel("Mean instantaneous semi-major-axis decay rate (m/day)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if output_path is None:
        output_path = (
            RUNTIME_DIRECTORY / "orbital_decay_solar_cycle_comparison.png"
        )
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Solar-cycle comparison graph saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def solar_cycle_phase_epochs() -> np.ndarray:
    """Map 0--100% onto the observed Cycle 25 minimum-to-maximum interval."""
    interval_seconds = int(
        (
            SOLAR_CYCLE_MAXIMUM_EPOCH - SOLAR_CYCLE_MINIMUM_EPOCH
        ).astype("timedelta64[s]").astype(np.int64)
    )
    offsets = np.rint(
        SOLAR_CYCLE_PHASE_PERCENTAGES * interval_seconds / 100.0
    ).astype("timedelta64[s]")
    return SOLAR_CYCLE_MINIMUM_EPOCH + offsets


def plot_solar_cycle_phase_surface(
    phases_percent: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """Plot uncapped instantaneous decay as a two-dimensional heatmap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects

    maximum_mean = float(np.nanmax(means))
    colour_ceiling = 10.0 * np.ceil(maximum_mean / 10.0)
    mean_colour_norm = Normalize(vmin=0.0, vmax=colour_ceiling)

    figure, axis = plt.subplots(figsize=(12, 8))
    heatmap = axis.pcolormesh(
        phases_percent,
        SAMPLE_HEIGHTS_KM,
        means.T,
        cmap="viridis",
        norm=mean_colour_norm,
        shading="auto",
        rasterized=True,
    )

    candidate_uncertainty_levels = np.array((1.0, 2.0, 5.0, 10.0, 15.0))
    uncertainty_levels = candidate_uncertainty_levels[
        (candidate_uncertainty_levels > np.nanmin(standard_deviations))
        & (candidate_uncertainty_levels < np.nanmax(standard_deviations))
    ]
    uncertainty_contours = axis.contour(
        phases_percent,
        SAMPLE_HEIGHTS_KM,
        standard_deviations.T,
        levels=uncertainty_levels,
        colors="white",
        linewidths=1.0,
        alpha=0.9,
    )
    contour_labels = axis.clabel(
        uncertainty_contours,
        inline=True,
        fontsize=9,
        fmt=lambda value: f"1 sigma = {value:g}",
    )
    for label in contour_labels:
        label.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.5, foreground="black"),
                path_effects.Normal(),
            ]
        )

    colourbar = figure.colorbar(heatmap, ax=axis, pad=0.025)
    colourbar.set_label("Mean instantaneous net decay rate (m/day)")
    colourbar_ticks = np.arange(0.0, colour_ceiling + 1.0, 20.0)
    colourbar.set_ticks(colourbar_ticks)

    axis.set_title(
        "Instantaneous orbit-averaged VELOX-C1 net orbital decay\n"
        "Observed Solar Cycle 25 minimum-to-maximum ascent; "
        f"drag + vector SRP, {MONTE_CARLO_RUNS} Monte Carlo realizations"
    )
    axis.set_xlabel("Solar-cycle phase (% of minimum-to-maximum interval)")
    axis.set_ylabel("Initial altitude (km)")
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(500.0, 700.0)
    axis.set_xticks(np.arange(0.0, 101.0, 20.0))
    axis.set_yticks(np.arange(500.0, 701.0, 50.0))
    figure.tight_layout()

    if output_path is None:
        output_path = (
            RUNTIME_DIRECTORY / "orbital_decay_solar_cycle_phase_heatmap.png"
        )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    print(f"Solar-cycle phase heatmap saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def run_start_period(task: tuple[str, bool]) -> tuple[np.datetime64, np.ndarray]:
    """Worker entry point for one independently propagated start period."""
    launch_text, use_cache = task
    launch_epoch = parse_launch_epoch(launch_text)
    return launch_epoch, run_monte_carlo(launch_epoch, use_cache=use_cache)


def run_instantaneous_start_period(
    task: tuple[str, bool],
) -> tuple[np.datetime64, np.ndarray]:
    """Worker entry point for one epoch-local solar-cycle evaluation."""
    launch_text, use_cache = task
    launch_epoch = parse_launch_epoch(launch_text)
    decay_rates = run_monte_carlo(
        launch_epoch,
        use_cache=use_cache,
        simulation_duration=INSTANTANEOUS_EVALUATION_DURATION,
        verbose=False,
    )
    return launch_epoch, decay_rates


def run_solar_phase_period(
    task: tuple[float, str, bool],
) -> tuple[float, np.datetime64, np.ndarray]:
    """Worker entry point for one epoch-local solar-phase evaluation."""
    phase_percent, launch_text, use_cache = task
    launch_epoch = parse_launch_epoch(launch_text)
    decay_rates = run_monte_carlo(
        launch_epoch,
        use_cache=use_cache,
        simulation_duration=INSTANTANEOUS_EVALUATION_DURATION,
        verbose=False,
    )
    return phase_percent, launch_epoch, decay_rates


def positive_worker_count(text: str) -> int:
    """Parse a strictly positive process count for argparse."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("worker count must be at least 1")
    return value


def parse_command_line() -> argparse.Namespace:
    """Return command-line options for single or batched propagation."""
    parser = argparse.ArgumentParser(
        description=(
            "Propagate VELOX-C1 decay from one or more UTC launch epochs. "
            "Repeat --start to queue independent periods."
        )
    )
    parser.add_argument(
        "--start",
        action="append",
        metavar="ISO_EPOCH",
        help=(
            "UTC launch epoch; repeat for multiple periods. If omitted, the "
            "program asks for one epoch interactively."
        ),
    )
    parser.add_argument(
        "--workers",
        type=positive_worker_count,
        default=1,
        help="number of independent start periods to run concurrently",
    )
    parser.add_argument(
        "--solar-cycle-comparison",
        action="store_true",
        help=(
            "automatically run the documented Solar Cycle 25 maximum, "
            "temporal midpoint, and minimum epochs as instantaneous "
            "orbit-averaged rates and overlay one graph"
        ),
    )
    parser.add_argument(
        "--solar-cycle-surface",
        action="store_true",
        help=(
            "create a 2D heatmap of instantaneous decay versus altitude and "
            "percentage through the observed Cycle 25 minimum-to-maximum "
            "interval"
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore and do not write the exact completed-run cache",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="save graphs without opening interactive plot windows",
    )
    # IPython/Jupyter adds kernel arguments such as ``-f kernel.json``. They
    # belong to the notebook runtime, not this model's command-line interface.
    arguments = parser.parse_args([] if _SOURCE_FILE is None else None)
    if arguments.solar_cycle_comparison and arguments.solar_cycle_surface:
        parser.error(
            "--solar-cycle-comparison cannot be combined with "
            "--solar-cycle-surface"
        )
    if (
        arguments.solar_cycle_comparison or arguments.solar_cycle_surface
    ) and arguments.start:
        parser.error(
            "solar-cycle graph modes cannot be combined with --start"
        )
    return arguments


def main() -> None:
    """Run one or more altitude sweeps and plot their uncertainties."""
    arguments = parse_command_line()
    if arguments.solar_cycle_surface:
        phase_epochs = solar_cycle_phase_epochs()
        phase_tasks = [
            (
                float(phase),
                np.datetime_as_string(epoch, unit="s"),
                not arguments.no_cache,
            )
            for phase, epoch in zip(
                SOLAR_CYCLE_PHASE_PERCENTAGES, phase_epochs
            )
        ]
        worker_count = min(arguments.workers, len(phase_tasks))
        print(
            "Evaluating instantaneous decay at "
            f"{len(phase_tasks)} observed solar-cycle phase epochs."
        )
        if worker_count > 1:
            print(f"Using {worker_count} worker processes.")
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count
                ) as executor:
                    phase_results = list(
                        executor.map(run_solar_phase_period, phase_tasks)
                    )
            except (OSError, RuntimeError) as error:
                print(
                    "Worker processes are unavailable; continuing "
                    f"sequentially ({error})."
                )
                phase_results = [
                    run_solar_phase_period(task) for task in phase_tasks
                ]
        else:
            phase_results = [
                run_solar_phase_period(task) for task in phase_tasks
            ]

        means = []
        standard_deviations = []
        for _, _, decay_rates in phase_results:
            mean, standard_deviation = summarize_decay_rates(decay_rates)
            means.append(mean)
            standard_deviations.append(standard_deviation)
        mean_surface = np.stack(means)
        uncertainty_surface = np.stack(standard_deviations)

        print("\nReference-phase instantaneous rates (mean +/- 1 sigma):")
        print("Phase   Date          500 km       600 km       700 km")
        for phase_index in (0, len(phase_results) // 2, len(phase_results) - 1):
            phase, epoch, _ = phase_results[phase_index]
            formatted_rates = "  ".join(
                f"{mean_surface[phase_index, altitude_index]:6.2f} +/- "
                f"{uncertainty_surface[phase_index, altitude_index]:5.2f}"
                for altitude_index in (0, 10, 20)
            )
            print(
                f"{phase:5.0f}%  "
                f"{np.datetime_as_string(epoch, unit='D')}  "
                f"{formatted_rates} m/day"
            )
        plot_solar_cycle_phase_surface(
            SOLAR_CYCLE_PHASE_PERCENTAGES,
            mean_surface,
            uncertainty_surface,
            show=not arguments.no_show,
        )
        return

    comparison_mode = arguments.solar_cycle_comparison
    if comparison_mode:
        launch_texts = [item[1] for item in SOLAR_CYCLE_COMPARISON_EPOCHS]
    else:
        launch_texts = arguments.start
    if not comparison_mode and not launch_texts:
        launch_texts = [
            input(
                "Launch epoch in UTC "
                "(YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS): "
            )
        ]

    # Normalize and validate all epochs before launching expensive workers.
    launch_texts = [
        np.datetime_as_string(parse_launch_epoch(text), unit="s")
        for text in launch_texts
    ]
    tasks = [(text, not arguments.no_cache) for text in launch_texts]
    run_period = (
        run_instantaneous_start_period
        if comparison_mode
        else run_start_period
    )
    worker_count = min(arguments.workers, len(tasks))
    if worker_count > 1:
        print(
            f"Running {len(tasks)} start periods with "
            f"{worker_count} worker processes."
        )
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(run_period, tasks))
        except (OSError, RuntimeError) as error:
            print(
                "Worker processes are unavailable; continuing sequentially "
                f"({error})."
            )
            results = [run_period(task) for task in tasks]
    else:
        results = [run_period(task) for task in tasks]

    if comparison_mode:
        comparison_results = []
        for (phase, _, colour), (launch_epoch, decay_rates) in zip(
            SOLAR_CYCLE_COMPARISON_EPOCHS, results
        ):
            mean, standard_deviation = summarize_decay_rates(decay_rates)
            print(
                f"\n{phase} results for "
                f"{np.datetime_as_string(launch_epoch, unit='s')} UTC"
            )
            print_summary(SAMPLE_HEIGHTS_KM, mean, standard_deviation)
            comparison_results.append(
                (phase, launch_epoch, mean, standard_deviation, colour)
            )
        plot_solar_cycle_comparison(
            comparison_results,
            show=not arguments.no_show,
        )
        return

    multiple_periods = len(results) > 1
    for launch_epoch, decay_rates in results:
        mean, standard_deviation = summarize_decay_rates(decay_rates)
        print(
            "\nResults for "
            f"{np.datetime_as_string(launch_epoch, unit='s')} UTC"
        )
        print_summary(SAMPLE_HEIGHTS_KM, mean, standard_deviation)
        output_path = None
        if multiple_periods:
            epoch_slug = np.datetime_as_string(
                launch_epoch, unit="s"
            ).replace(":", "-")
            output_path = (
                RUNTIME_DIRECTORY / f"orbital_decay_rate_{epoch_slug}.png"
            )
        plot_decay_rates(
            SAMPLE_HEIGHTS_KM,
            mean,
            standard_deviation,
            launch_epoch,
            output_path=output_path,
            show=not arguments.no_show and not multiple_periods,
        )


if __name__ == "__main__":
    main()
