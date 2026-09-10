"""Plotting functions used by the configurable desktop workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import model


EXPORT_PNG_DPI = 600
QUICK_DEMO_PNG_DPI = EXPORT_PNG_DPI
FIGURE_SIZE_INCHES = (10.0, 10.0)


def plot_decay_rates(
    heights_km: np.ndarray,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    launch_epoch: np.datetime64,
    output_path: Path,
    show: bool = False,
    spacecraft_label: str = "VELOX-C1",
    quick_demo: bool = False,
) -> None:
    """Render an altitude line graph for selected simulation parameters."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    smooth_heights, smooth_mean, lower_bound, upper_bound = (
        model.regression_trend_with_uncertainty(
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
    demo_note = "\nQUICK DEMO — approximate 30-day propagation step" if quick_demo else ""
    axis.set_title(
        f"{spacecraft_label} NRLMSIS/Sentman + vector-SRP decay rate "
        f"({model.MONTE_CARLO_RUNS} realizations)\n"
        f"One-year propagation from {launch_day} UTC{demo_note}"
    )
    axis.set_xlabel("Initial altitude (km)")
    axis.set_ylabel("Mean semi-major-axis decay rate (m/day)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=EXPORT_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    print(f"Graph saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_solar_cycle_phase_surface(
    phases_percent: np.ndarray,
    heights_km: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
    interval_start: np.datetime64,
    interval_end: np.datetime64,
    output_path: Path,
    show: bool = False,
    spacecraft_label: str = "VELOX-C1",
    quick_demo: bool = False,
) -> None:
    """Render an uncapped phase/altitude heat map for a selected interval."""
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    maximum_mean = float(np.nanmax(means))
    colour_ceiling = max(10.0, 10.0 * np.ceil(maximum_mean / 10.0))
    mean_colour_norm = Normalize(vmin=0.0, vmax=colour_ceiling)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)
    heatmap = axis.pcolormesh(
        phases_percent,
        heights_km,
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
    if uncertainty_levels.size:
        uncertainty_contours = axis.contour(
            phases_percent,
            heights_km,
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
    tick_step = 20.0 if colour_ceiling >= 40.0 else 5.0
    colourbar.set_ticks(
        np.arange(0.0, colour_ceiling + 0.5 * tick_step, tick_step)
    )

    start_day = np.datetime_as_string(interval_start, unit="D")
    end_day = np.datetime_as_string(interval_end, unit="D")
    is_reference_interval = (
        interval_start == model.SOLAR_CYCLE_MINIMUM_EPOCH
        and interval_end == model.SOLAR_CYCLE_MAXIMUM_EPOCH
    )
    if is_reference_interval:
        interval_description = "Observed Solar Cycle 25 minimum-to-maximum ascent"
        phase_label = "Solar-cycle phase (% of minimum-to-maximum interval)"
    else:
        interval_description = (
            f"Selected {start_day} to {end_day} interval "
            "(Cycle 25 ascent duration)"
        )
        phase_label = "Selected-interval phase (%)"
    demo_note = "\nQUICK DEMO — approximate 8-realization ensemble" if quick_demo else ""
    axis.set_title(
        f"Instantaneous orbit-averaged {spacecraft_label} net orbital decay\n"
        f"{interval_description}; drag + vector SRP, "
        f"{model.MONTE_CARLO_RUNS} Monte Carlo realizations{demo_note}"
    )
    axis.set_xlabel(phase_label)
    axis.set_ylabel("Initial altitude (km)")
    axis.set_xlim(0.0, 100.0)
    altitude_span = float(np.ptp(heights_km))
    altitude_padding = max(
        5.0, altitude_span / max(2 * (len(heights_km) - 1), 1)
    )
    axis.set_ylim(
        float(np.min(heights_km) - altitude_padding),
        float(np.max(heights_km) + altitude_padding),
    )
    axis.set_xticks(np.arange(0.0, 101.0, 20.0))
    if len(heights_km) <= 11:
        axis.set_yticks(heights_km)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=EXPORT_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    print(f"Phase heat map saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


__all__ = [
    "EXPORT_PNG_DPI",
    "FIGURE_SIZE_INCHES",
    "QUICK_DEMO_PNG_DPI",
    "plot_decay_rates",
    "plot_solar_cycle_phase_surface",
]
