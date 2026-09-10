"""VELOX-C1 orbital-decay simulation package."""

from .config import (
    ALTITUDE_SAMPLE_COUNT,
    PlotKind,
    SimulationRequest,
    SpacecraftParameters,
    VELOX_C1_DEFAULTS,
    latest_supported_launch_date,
    parse_altitude_bounds,
    parse_altitudes,
)

__all__ = [
    "ALTITUDE_SAMPLE_COUNT",
    "PlotKind",
    "SimulationRequest",
    "SpacecraftParameters",
    "VELOX_C1_DEFAULTS",
    "latest_supported_launch_date",
    "parse_altitude_bounds",
    "parse_altitudes",
]
__version__ = "2.0.0"
