"""Validated user-facing configuration for the simulation workflows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np

from . import model


PlotKind = Literal["line", "heatmap"]
ALTITUDE_SAMPLE_COUNT = 20


def _finite_float(text: str, field_name: str) -> float:
    """Parse finite numeric text and provide a field-specific error message."""
    try:
        value = float(text.strip())
    except ValueError as error:
        raise ValueError(f"{field_name} must be a number.") from error
    if not np.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")
    return value


def parse_altitudes(text: str) -> tuple[float, ...]:
    """Parse one altitude, a comma-separated list, or start:stop:step.

    The range syntax is inclusive, so ``500:700:10`` selects the complete
    500--700 km grid at 10 km spacing.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Enter at least one initial altitude.")

    if ":" in cleaned:
        parts = [part.strip() for part in cleaned.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "Altitude ranges must use start:stop:step, for example "
                "500:700:10."
            )
        try:
            start, stop, step = (float(part) for part in parts)
        except ValueError as error:
            raise ValueError("Altitude values must be numbers in kilometres.") from error
        if step <= 0.0:
            raise ValueError("The altitude increment must be positive.")
        if stop < start:
            raise ValueError("The upper altitude must not be below the lower altitude.")
        count = int(np.floor((stop - start) / step + 1e-10)) + 1
        values = start + step * np.arange(count, dtype=float)
        if not np.isclose(values[-1], stop, rtol=0.0, atol=1e-8):
            raise ValueError("The altitude increment must land exactly on the upper bound.")
    else:
        try:
            values = np.asarray(
                [float(part.strip()) for part in cleaned.split(",")],
                dtype=float,
            )
        except ValueError as error:
            raise ValueError("Altitude values must be numbers in kilometres.") from error

    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("All altitude values must be finite numbers.")
    values = np.unique(values)
    if np.any(values < 500.0) or np.any(values > 700.0):
        raise ValueError("This model is configured for altitudes from 500 to 700 km.")
    return tuple(float(value) for value in values)


def parse_altitude_bounds(
    lower_text: str,
    upper_text: str,
    sample_count: int = ALTITUDE_SAMPLE_COUNT,
) -> tuple[float, ...]:
    """Return evenly spaced, inclusive altitudes between two GUI bounds."""
    lower = _finite_float(lower_text, "Lower altitude")
    upper = _finite_float(upper_text, "Upper altitude")
    if lower < 500.0 or upper > 700.0:
        raise ValueError("Altitude bounds must remain between 500 and 700 km.")
    if upper <= lower:
        raise ValueError("Upper altitude must be greater than lower altitude.")
    if sample_count < 2:
        raise ValueError("At least two altitude samples are required.")
    return tuple(
        float(value) for value in np.linspace(lower, upper, sample_count)
    )


@lru_cache(maxsize=1)
def latest_geomagnetic_epoch() -> np.datetime64:
    """Return the last three-hour epoch covered by PyMSIS's local Ap data."""
    space_weather_path = model.space_weather_file_path()

    ap_fields = tuple(f"AP{index}" for index in range(1, 9))
    latest_day: np.datetime64 | None = None
    with space_weather_path.open(newline="") as file:
        for row in csv.DictReader(file):
            if not all(row.get(field, "").strip() for field in ap_fields):
                continue
            try:
                day = np.datetime64(row["DATE"], "D")
            except (KeyError, ValueError):
                continue
            if latest_day is None or day > latest_day:
                latest_day = day

    if latest_day is None:
        raise RuntimeError(
            "The PyMSIS space-weather file contains no usable geomagnetic data."
        )
    return latest_day.astype("datetime64[s]") + np.timedelta64(21, "h")


@lru_cache(maxsize=2)
def latest_supported_launch_epoch(plot_kind: PlotKind) -> np.datetime64:
    """Return the latest launch epoch whose complete graph has Ap coverage."""
    if plot_kind == "line":
        reference_epoch = np.datetime64("2000-01-01T00:00:00", "s")
        dates, _ = model.build_time_grid(reference_epoch)
        required_offset = dates[-1] - reference_epoch
    elif plot_kind == "heatmap":
        required_offset = (
            model.SOLAR_CYCLE_MAXIMUM_EPOCH
            - model.SOLAR_CYCLE_MINIMUM_EPOCH
        )
    else:
        raise ValueError("Plot type must be either line or heatmap.")
    return latest_geomagnetic_epoch() - required_offset


def latest_supported_launch_date(plot_kind: PlotKind) -> np.datetime64:
    """Return the latest supported GUI date for one graph type."""
    return latest_supported_launch_epoch(plot_kind).astype("datetime64[D]")


def validate_launch_date(
    launch_epoch: np.datetime64,
    plot_kind: PlotKind,
) -> None:
    """Reject dates whose requested graph extends beyond local Ap coverage."""
    latest_epoch = latest_supported_launch_epoch(plot_kind)
    if launch_epoch > latest_epoch:
        graph_name = "line graph" if plot_kind == "line" else "heat map"
        latest_date = np.datetime_as_string(
            latest_supported_launch_date(plot_kind), unit="D"
        )
        raise ValueError(
            f"Launch date is too late for a complete {graph_name}. "
            f"The latest supported date is {latest_date}."
        )


@dataclass(frozen=True, slots=True)
class SpacecraftParameters:
    """Editable spacecraft and starting-orbit values used by the model."""

    mass_kg: float = float(model.SATELLITE_MASS)
    aerodynamic_area_m2: float = float(model.CROSS_SECTIONAL_AREA)
    body_dimensions_m: tuple[float, float, float] = tuple(
        float(value) for value in model.VELOX_BODY_DIMENSIONS
    )
    solar_radiation_area_m2: float = float(model.SOLAR_RADIATION_AREA)
    solar_reflection_factor: float = float(model.SOLAR_REFLECTION_FACTOR)
    eccentricity: float = float(model.ECCENTRICITY)
    inclination_deg: float = float(model.INCLINATION_DEG)
    raan_deg: float = float(model.RAAN_DEG)
    argument_of_perigee_deg: float = float(model.ARGUMENT_OF_PERIGEE_DEG)

    @classmethod
    def from_text(
        cls,
        mass_kg: str,
        aerodynamic_area_m2: str,
        body_x_m: str,
        body_y_m: str,
        body_z_m: str,
        solar_radiation_area_m2: str,
        solar_reflection_factor: str,
        eccentricity: str,
        inclination_deg: str,
        raan_deg: str,
        argument_of_perigee_deg: str,
    ) -> "SpacecraftParameters":
        """Parse, validate, and normalize spacecraft values from text fields."""
        values = cls(
            mass_kg=_finite_float(mass_kg, "Mass"),
            aerodynamic_area_m2=_finite_float(
                aerodynamic_area_m2, "Aerodynamic area"
            ),
            body_dimensions_m=(
                _finite_float(body_x_m, "Body X dimension"),
                _finite_float(body_y_m, "Body Y dimension"),
                _finite_float(body_z_m, "Body Z dimension"),
            ),
            solar_radiation_area_m2=_finite_float(
                solar_radiation_area_m2, "Solar-radiation area"
            ),
            solar_reflection_factor=_finite_float(
                solar_reflection_factor, "Solar reflection factor"
            ),
            eccentricity=_finite_float(eccentricity, "Eccentricity"),
            inclination_deg=_finite_float(inclination_deg, "Inclination"),
            raan_deg=_finite_float(raan_deg, "RAAN"),
            argument_of_perigee_deg=_finite_float(
                argument_of_perigee_deg, "Argument of perigee"
            ),
        )
        if values.mass_kg <= 0.0:
            raise ValueError("Mass must be greater than zero.")
        if values.aerodynamic_area_m2 <= 0.0:
            raise ValueError("Aerodynamic area must be greater than zero.")
        if values.solar_radiation_area_m2 <= 0.0:
            raise ValueError("Solar-radiation area must be greater than zero.")
        if any(dimension <= 0.0 for dimension in values.body_dimensions_m):
            raise ValueError("Every body dimension must be greater than zero.")
        if not 0.0 <= values.solar_reflection_factor <= 1.0:
            raise ValueError("Solar reflection factor must be between 0 and 1.")
        if not 0.0 <= values.eccentricity < 1.0:
            raise ValueError("Eccentricity must be at least 0 and below 1.")
        if not 0.0 <= values.inclination_deg <= 180.0:
            raise ValueError("Inclination must be between 0 and 180 degrees.")
        return cls(
            mass_kg=values.mass_kg,
            aerodynamic_area_m2=values.aerodynamic_area_m2,
            body_dimensions_m=values.body_dimensions_m,
            solar_radiation_area_m2=values.solar_radiation_area_m2,
            solar_reflection_factor=values.solar_reflection_factor,
            eccentricity=values.eccentricity,
            inclination_deg=values.inclination_deg,
            raan_deg=values.raan_deg % 360.0,
            argument_of_perigee_deg=values.argument_of_perigee_deg % 360.0,
        )


VELOX_C1_DEFAULTS = SpacecraftParameters()


@dataclass(frozen=True, slots=True)
class SimulationRequest:
    """One validated line-graph or heat-map request from the GUI."""

    launch_date: str
    altitudes_km: tuple[float, ...]
    plot_kind: PlotKind
    use_cache: bool = True
    quick_demo: bool = False
    spacecraft: SpacecraftParameters = VELOX_C1_DEFAULTS

    @classmethod
    def from_text(
        cls,
        launch_date: str,
        altitude_text: str,
        plot_kind: str,
        use_cache: bool = True,
        quick_demo: bool = False,
        spacecraft: SpacecraftParameters = VELOX_C1_DEFAULTS,
    ) -> "SimulationRequest":
        """Create a validated request from a flexible altitude expression."""
        if plot_kind not in ("line", "heatmap"):
            raise ValueError("Plot type must be either line or heatmap.")
        altitudes = parse_altitudes(altitude_text)
        if plot_kind == "heatmap" and len(altitudes) < 2:
            raise ValueError("A heat map requires at least two initial altitudes.")

        launch_epoch = model.parse_launch_epoch(launch_date)
        validate_launch_date(launch_epoch, plot_kind)
        normalized_date = np.datetime_as_string(launch_epoch, unit="s")
        return cls(
            launch_date=normalized_date,
            altitudes_km=altitudes,
            plot_kind=plot_kind,
            use_cache=bool(use_cache),
            quick_demo=bool(quick_demo),
            spacecraft=spacecraft,
        )

    @classmethod
    def from_bounds(
        cls,
        launch_date: str,
        lower_altitude: str,
        upper_altitude: str,
        plot_kind: str,
        use_cache: bool = True,
        quick_demo: bool = False,
        spacecraft: SpacecraftParameters = VELOX_C1_DEFAULTS,
    ) -> "SimulationRequest":
        """Build a request from the GUI's lower and upper altitude fields."""
        if plot_kind not in ("line", "heatmap"):
            raise ValueError("Plot type must be either line or heatmap.")
        altitudes = parse_altitude_bounds(lower_altitude, upper_altitude)
        launch_epoch = model.parse_launch_epoch(launch_date)
        validate_launch_date(launch_epoch, plot_kind)
        return cls(
            launch_date=np.datetime_as_string(launch_epoch, unit="s"),
            altitudes_km=altitudes,
            plot_kind=plot_kind,
            use_cache=bool(use_cache),
            quick_demo=bool(quick_demo),
            spacecraft=spacecraft,
        )
