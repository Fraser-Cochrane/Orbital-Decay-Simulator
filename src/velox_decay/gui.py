"""Tkinter desktop interface for the VELOX-C1 orbital decay model."""

from __future__ import annotations

import calendar
from datetime import date
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser

import numpy as np
from PIL import Image, ImageTk

from . import model
from .config import (
    SimulationRequest,
    SpacecraftParameters,
    VELOX_C1_DEFAULTS,
    latest_supported_launch_date,
)
from .workflows import RunResult, run_request


OUTPUT_DIRECTORY = model.RUNTIME_DIRECTORY


class OrbitalDecayApp(tk.Tk):
    """Responsive desktop GUI for line-graph and heat-map workflows."""

    BACKGROUND = "#edf3f1"
    CARD = "#fbfdfc"
    CARD_ALT = "#f4f8f7"
    INK = "#223638"
    MUTED = "#697b7b"
    ACCENT = "#4b7772"
    ACCENT_ACTIVE = "#38615d"
    ACCENT_SOFT = "#dce9e6"
    INPUT = "#f7faf9"
    BORDER = "#d3e0dd"
    PREVIEW = "#ffffff"

    def __init__(self) -> None:
        """Initialize application state, styles, controls, and help text."""
        super().__init__()
        self.title("VELOX-C1 Orbital Decay")
        self.geometry("1360x820")
        self.minsize(1130, 760)
        self.configure(background=self.BACKGROUND)

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._last_result: RunResult | None = None
        self._preview_image: ImageTk.PhotoImage | None = None
        self._preview_source: Path | None = None
        self._preview_resize_job: str | None = None
        self._spacecraft_window: tk.Toplevel | None = None
        self._active_plot_kind = "line"
        self._line_date = "2024-01-01"
        self._heatmap_date = "2019-12-01"

        # Use a covered historical epoch rather than today's date. A one-year
        # run beginning near the end of the local CelesTrak daily-Ap record
        # would otherwise extend beyond the data needed by NRLMSIS.
        self.launch_year = tk.StringVar(value="2024")
        self.launch_month = tk.StringVar(value="01")
        self.launch_day = tk.StringVar(value="01")
        self.lower_altitude = tk.StringVar(value="500")
        self.upper_altitude = tk.StringVar(value="700")
        self.plot_kind = tk.StringVar(value="line")
        self.use_cache = tk.BooleanVar(value=True)
        self.quick_demo = tk.BooleanVar(value=False)
        self.progress_value = tk.DoubleVar(value=0.0)
        self.status = tk.StringVar(value="Ready")
        self.spacecraft_summary = tk.StringVar()
        self.spacecraft_values = {
            "mass": tk.StringVar(value="123.0"),
            "drag_area": tk.StringVar(value="0.52"),
            "body_x": tk.StringVar(value="0.615"),
            "body_y": tk.StringVar(value="0.608"),
            "body_z": tk.StringVar(value="0.848"),
            "srp_area": tk.StringVar(value="0.52"),
            "reflection": tk.StringVar(value="0.5"),
            "eccentricity": tk.StringVar(value="0.0009024"),
            "inclination": tk.StringVar(value="15.0"),
            "raan": tk.StringVar(value="0.0"),
            "perigee": tk.StringVar(value="0.0"),
        }
        for value in self.spacecraft_values.values():
            value.trace_add("write", self._update_spacecraft_summary)
        self._update_spacecraft_summary()

        self._configure_style()
        self._build_interface()
        self._update_mode_help()

    def _configure_style(self) -> None:
        """Define the application palette and reusable ttk widget styles."""
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        self.option_add("*TCombobox*Listbox.background", self.INPUT)
        self.option_add("*TCombobox*Listbox.foreground", self.INK)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT_SOFT)
        self.option_add("*TCombobox*Listbox.selectForeground", self.INK)
        style.configure("Page.TFrame", background=self.BACKGROUND)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure("Inset.TFrame", background=self.CARD_ALT)
        style.configure(
            "Panel.TFrame",
            background=self.CARD,
            bordercolor=self.BORDER,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Title.TLabel",
            background=self.BACKGROUND,
            foreground=self.INK,
            font=("Helvetica Neue", 27, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=self.BACKGROUND,
            foreground=self.MUTED,
            font=("Helvetica Neue", 11),
        )
        style.configure(
            "Badge.TLabel",
            background=self.ACCENT_SOFT,
            foreground=self.ACCENT_ACTIVE,
            font=("Helvetica Neue", 9, "bold"),
            padding=(12, 7),
        )
        style.configure(
            "PanelTitle.TLabel",
            background=self.CARD,
            foreground=self.INK,
            font=("Helvetica Neue", 18, "bold"),
        )
        style.configure(
            "PanelSubtitle.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Helvetica Neue", 10),
        )
        style.configure(
            "Section.TLabel",
            background=self.CARD,
            foreground=self.ACCENT,
            font=("Helvetica Neue", 9, "bold"),
        )
        style.configure(
            "Field.TLabel",
            background=self.CARD,
            foreground=self.INK,
            font=("Helvetica Neue", 11, "bold"),
        )
        style.configure(
            "Help.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Helvetica Neue", 10),
        )
        style.configure(
            "InputCaption.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("Helvetica Neue", 9),
        )
        style.configure(
            "Input.TEntry",
            fieldbackground=self.INPUT,
            foreground=self.INK,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=(7, 6),
        )
        style.map(
            "Input.TEntry",
            bordercolor=[("focus", self.ACCENT)],
            lightcolor=[("focus", self.ACCENT)],
            darkcolor=[("focus", self.ACCENT)],
        )
        style.configure(
            "Date.TCombobox",
            fieldbackground=self.INPUT,
            background=self.INPUT,
            foreground=self.INK,
            arrowcolor=self.ACCENT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=(7, 5),
        )
        style.map(
            "Date.TCombobox",
            bordercolor=[("focus", self.ACCENT)],
            lightcolor=[("focus", self.ACCENT)],
            darkcolor=[("focus", self.ACCENT)],
            fieldbackground=[("readonly", self.INPUT)],
            foreground=[("readonly", self.INK)],
        )
        style.configure(
            "Primary.TButton",
            background=self.ACCENT,
            foreground="white",
            font=("Helvetica Neue", 11, "bold"),
            padding=(18, 11),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "Primary.TButton",
            background=[
                ("active", self.ACCENT_ACTIVE),
                ("disabled", "#a7b7b5"),
            ],
        )
        style.configure(
            "Secondary.TButton",
            background="#e2ebe9",
            foreground=self.INK,
            bordercolor=self.BORDER,
            padding=(15, 10),
            relief="flat",
        )
        style.map("Secondary.TButton", background=[("active", "#d5e2df")])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#d8e3e1",
            background=self.ACCENT,
            bordercolor="#d8e3e1",
            lightcolor=self.ACCENT,
            darkcolor=self.ACCENT,
            thickness=8,
        )
        style.configure(
            "Choice.TRadiobutton",
            background=self.CARD_ALT,
            foreground=self.INK,
            padding=(12, 8),
            indicatorcolor=self.CARD,
            indicatormargin=5,
        )
        style.map(
            "Choice.TRadiobutton",
            background=[("active", self.ACCENT_SOFT)],
            indicatorcolor=[("selected", self.ACCENT)],
        )
        style.configure(
            "Option.TCheckbutton",
            background=self.CARD,
            foreground=self.INK,
            padding=(0, 3),
            indicatorcolor=self.INPUT,
        )
        style.map(
            "Option.TCheckbutton",
            foreground=[("active", self.ACCENT_ACTIVE)],
            indicatorcolor=[("selected", self.ACCENT)],
        )
        style.configure(
            "Preview.TLabel",
            background=self.PREVIEW,
            foreground=self.MUTED,
            font=("Helvetica Neue", 11),
            anchor="center",
        )

    def _build_interface(self) -> None:
        """Construct the responsive configuration and graph-preview panels."""
        page = ttk.Frame(self, style="Page.TFrame", padding=(36, 22, 36, 24))
        page.pack(fill="both", expand=True)

        header = ttk.Frame(page, style="Page.TFrame")
        header.pack(fill="x", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        title_block = ttk.Frame(header, style="Page.TFrame")
        title_block.grid(row=0, column=0, sticky="w")
        ttk.Label(
            title_block,
            text="VELOX-C1 Orbital Decay",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            title_block,
            text="NRLMSIS 2.1 · Sentman/DRIA · vector solar-radiation pressure",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            header,
            text="RESEARCH SIMULATOR",
            style="Badge.TLabel",
        ).grid(row=0, column=1, sticky="e")

        body = ttk.Frame(page, style="Page.TFrame")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, minsize=535)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        card = ttk.Frame(body, style="Panel.TFrame", padding=(26, 18, 26, 18))
        card.grid(row=0, column=0, sticky="nsew")
        card.columnconfigure(1, weight=1)

        preview_card = ttk.Frame(
            body, style="Panel.TFrame", padding=(8, 8, 8, 8)
        )
        preview_card.grid(row=0, column=1, sticky="nsew", padx=(20, 0))
        self.preview_label = ttk.Label(
            preview_card,
            text="Graph preview\n\nChoose the simulation settings and generate a graph.",
            style="Preview.TLabel",
            justify="center",
        )
        self.preview_label.pack(fill="both", expand=True)
        self.preview_label.bind("<Configure>", self._schedule_preview_resize)

        ttk.Label(card, text="GRAPH", style="Section.TLabel").grid(
            row=0, column=0, sticky="nw", padx=(0, 24), pady=(2, 9)
        )
        output_choices = ttk.Frame(card, style="Inset.TFrame", padding=(3, 3))
        output_choices.grid(row=0, column=1, sticky="w", pady=(0, 9))
        ttk.Radiobutton(
            output_choices,
            text="Line graph",
            variable=self.plot_kind,
            value="line",
            command=self._update_mode_help,
            style="Choice.TRadiobutton",
        ).pack(side="left")
        ttk.Radiobutton(
            output_choices,
            text="Heat map",
            variable=self.plot_kind,
            value="heatmap",
            command=self._update_mode_help,
            style="Choice.TRadiobutton",
        ).pack(side="left")

        ttk.Label(card, text="Launch date", style="Field.TLabel").grid(
            row=1, column=0, sticky="nw", padx=(0, 24), pady=(3, 7)
        )
        date_controls = ttk.Frame(card, style="Card.TFrame")
        date_controls.grid(row=1, column=1, sticky="w", pady=(0, 7))
        date_fields = (
            ("Year", self.launch_year, 7),
            ("Month", self.launch_month, 5),
            ("Day", self.launch_day, 5),
        )
        self.date_boxes: list[ttk.Combobox] = []
        for column, (label, variable, width) in enumerate(date_fields):
            field = ttk.Frame(date_controls, style="Card.TFrame")
            field.grid(row=0, column=column, sticky="w", padx=(0, 12))
            ttk.Label(field, text=label, style="InputCaption.TLabel").pack(
                anchor="w", pady=(0, 3)
            )
            box = ttk.Combobox(
                field,
                textvariable=variable,
                width=width,
                state="readonly",
                style="Date.TCombobox",
            )
            box.pack(anchor="w")
            box.bind("<<ComboboxSelected>>", self._date_part_changed)
            self.date_boxes.append(box)
        self.year_box, self.month_box, self.day_box = self.date_boxes
        self.date_help = ttk.Label(card, style="Help.TLabel", wraplength=470)
        self.date_help.grid(row=2, column=1, sticky="w", pady=(0, 11))

        ttk.Label(card, text="Altitude range", style="Field.TLabel").grid(
            row=3, column=0, sticky="nw", padx=(0, 24), pady=(3, 7)
        )
        altitude_controls = ttk.Frame(card, style="Card.TFrame")
        altitude_controls.grid(row=3, column=1, sticky="w", pady=(0, 7))
        altitude_fields = (
            ("Lower", self.lower_altitude),
            ("Upper", self.upper_altitude),
        )
        for column, (label, variable) in enumerate(altitude_fields):
            field = ttk.Frame(altitude_controls, style="Card.TFrame")
            field.grid(row=0, column=column, sticky="w", padx=(0, 18))
            ttk.Label(field, text=label, style="InputCaption.TLabel").pack(
                anchor="w", pady=(0, 3)
            )
            value_row = ttk.Frame(field, style="Card.TFrame")
            value_row.pack(anchor="w")
            ttk.Entry(
                value_row,
                textvariable=variable,
                width=9,
                style="Input.TEntry",
            ).pack(side="left")
            ttk.Label(value_row, text="km", style="Help.TLabel").pack(
                side="left", padx=(6, 0)
            )
        ttk.Label(
            card,
            text=(
                "500–700 km only. 20 evenly spaced values, including both "
                "bounds, will be evaluated."
            ),
            style="Help.TLabel",
            wraplength=470,
        ).grid(row=4, column=1, sticky="w", pady=(0, 11))

        ttk.Separator(card).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 11)
        )

        ttk.Label(card, text="SPACECRAFT", style="Section.TLabel").grid(
            row=6, column=0, sticky="nw", padx=(0, 24), pady=(2, 7)
        )
        ttk.Button(
            card,
            text="Edit spacecraft and orbit…",
            style="Secondary.TButton",
            command=self._open_spacecraft_settings,
        ).grid(row=6, column=1, sticky="w", pady=(0, 7))
        ttk.Label(
            card,
            textvariable=self.spacecraft_summary,
            style="Help.TLabel",
            wraplength=430,
        ).grid(row=7, column=1, sticky="w", pady=(0, 11))

        ttk.Separator(card).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(0, 10)
        )

        ttk.Checkbutton(
            card,
            text="Reuse exact-match completed-run cache",
            variable=self.use_cache,
            style="Option.TCheckbutton",
        ).grid(row=9, column=1, sticky="w", pady=(0, 3))

        ttk.Checkbutton(
            card,
            text="Quick demo (approximate)",
            variable=self.quick_demo,
            style="Option.TCheckbutton",
        ).grid(row=10, column=1, sticky="w", pady=(0, 3))
        ttk.Label(
            card,
            text=(
                "Demonstration only: the line graph uses 8 realizations and "
                "a 30-day step; the heat map uses 8 realizations. Demo cache "
                "entries are isolated from full-accuracy results."
            ),
            style="Help.TLabel",
            wraplength=470,
        ).grid(row=11, column=1, sticky="w", pady=(0, 11))

        separator = ttk.Separator(card)
        separator.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        self.progress = ttk.Progressbar(
            card,
            variable=self.progress_value,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.grid(row=13, column=0, columnspan=2, sticky="ew")
        status_row = ttk.Frame(card, style="Card.TFrame")
        status_row.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(6, 12))
        status_row.columnconfigure(0, weight=1)
        ttk.Label(
            status_row,
            textvariable=self.status,
            style="Help.TLabel",
            wraplength=430,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        self.percent_label = ttk.Label(status_row, text="0%", style="Help.TLabel")
        self.percent_label.grid(row=0, column=1, sticky="e")

        buttons = ttk.Frame(card, style="Card.TFrame")
        buttons.grid(row=15, column=0, columnspan=2, sticky="e")
        self.open_button = ttk.Button(
            buttons,
            text="Open graph",
            style="Secondary.TButton",
            command=self._open_graph,
            state="disabled",
        )
        self.open_button.pack(side="left", padx=(0, 10))
        self.run_button = ttk.Button(
            buttons,
            text="Generate graph",
            style="Primary.TButton",
            command=self._start_run,
        )
        self.run_button.pack(side="left")

    def _update_mode_help(self) -> None:
        """Store per-mode dates and explain the selected graph workflow."""
        selected_mode = self.plot_kind.get()
        if selected_mode != self._active_plot_kind:
            if self._active_plot_kind == "line":
                self._line_date = self._selected_date_text(self._active_plot_kind)
            else:
                self._heatmap_date = self._selected_date_text(
                    self._active_plot_kind
                )
            self._set_date_parts(
                self._line_date if selected_mode == "line" else self._heatmap_date
            )
            self._active_plot_kind = selected_mode

        self._refresh_date_choices()
        latest_date = np.datetime_as_string(
            latest_supported_launch_date(selected_mode), unit="D"
        )
        if selected_mode == "line":
            self.date_help.configure(
                text=(
                    "Start of the one-year propagation. The latest supported "
                    f"launch date is {latest_date}."
                )
            )
        else:
            self.date_help.configure(
                text=(
                    "Launch date at the 0% phase reference. The map retains the "
                    "Cycle 25 ascent duration and 21 phase samples. The latest "
                    f"supported launch date is {latest_date}."
                )
            )

    def _date_part_changed(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        """Refresh dependent month and day choices after a date change."""
        self._refresh_date_choices()

    def _set_date_parts(self, date_text: str) -> None:
        """Populate the date selectors from an ISO calendar date string."""
        selected = date.fromisoformat(date_text[:10])
        self.launch_year.set(f"{selected.year:04d}")
        self.launch_month.set(f"{selected.month:02d}")
        self.launch_day.set(f"{selected.day:02d}")

    def _refresh_date_choices(self) -> None:
        """Constrain calendar selectors to valid, data-supported dates."""
        maximum = date.fromisoformat(
            np.datetime_as_string(
                latest_supported_launch_date(self.plot_kind.get()), unit="D"
            )
        )
        minimum_year = 1961

        try:
            selected_year = int(self.launch_year.get())
        except ValueError:
            selected_year = maximum.year
        selected_year = min(max(selected_year, minimum_year), maximum.year)
        years = tuple(str(year) for year in range(minimum_year, maximum.year + 1))
        self.year_box.configure(values=years)
        self.launch_year.set(f"{selected_year:04d}")

        month_limit = maximum.month if selected_year == maximum.year else 12
        months = tuple(f"{month:02d}" for month in range(1, month_limit + 1))
        try:
            selected_month = int(self.launch_month.get())
        except ValueError:
            selected_month = 1
        selected_month = min(max(selected_month, 1), month_limit)
        self.month_box.configure(values=months)
        self.launch_month.set(f"{selected_month:02d}")

        day_limit = calendar.monthrange(selected_year, selected_month)[1]
        if selected_year == maximum.year and selected_month == maximum.month:
            day_limit = min(day_limit, maximum.day)
        days = tuple(f"{day:02d}" for day in range(1, day_limit + 1))
        try:
            selected_day = int(self.launch_day.get())
        except ValueError:
            selected_day = 1
        selected_day = min(max(selected_day, 1), day_limit)
        self.day_box.configure(values=days)
        self.launch_day.set(f"{selected_day:02d}")

    def _selected_date_text(self, plot_kind: str | None = None) -> str:
        """Return the validated selected date in ISO format."""
        try:
            selected = date(
                int(self.launch_year.get()),
                int(self.launch_month.get()),
                int(self.launch_day.get()),
            )
        except ValueError as error:
            raise ValueError("Select a valid launch year, month, and day.") from error
        maximum = date.fromisoformat(
            np.datetime_as_string(
                latest_supported_launch_date(plot_kind or self.plot_kind.get()),
                unit="D",
            )
        )
        if selected > maximum:
            raise ValueError(
                "Launch date is too late for the selected graph. "
                f"The latest supported date is {maximum.isoformat()}."
            )
        return selected.isoformat()

    def _parse_spacecraft_parameters(self) -> SpacecraftParameters:
        """Validate the spacecraft editor's text variables as model inputs."""
        values = self.spacecraft_values
        return SpacecraftParameters.from_text(
            values["mass"].get(),
            values["drag_area"].get(),
            values["body_x"].get(),
            values["body_y"].get(),
            values["body_z"].get(),
            values["srp_area"].get(),
            values["reflection"].get(),
            values["eccentricity"].get(),
            values["inclination"].get(),
            values["raan"].get(),
            values["perigee"].get(),
        )

    def _update_spacecraft_summary(self, *_: object) -> None:
        """Summarize whether the editor contains VELOX-C1 or custom values."""
        try:
            parameters = self._parse_spacecraft_parameters()
        except (ValueError, AttributeError):
            self.spacecraft_summary.set("Custom values — pending validation")
            return
        if parameters == VELOX_C1_DEFAULTS:
            self.spacecraft_summary.set(
                "VELOX-C1 defaults — 123 kg, 0.52 m² aerodynamic/SRP area"
            )
        else:
            self.spacecraft_summary.set(
                f"Custom — {parameters.mass_kg:g} kg, "
                f"{parameters.aerodynamic_area_m2:g} m² aerodynamic area"
            )

    def _reset_spacecraft_defaults(self) -> None:
        """Restore the documented VELOX-C1 spacecraft and orbit values."""
        defaults = {
            "mass": "123.0",
            "drag_area": "0.52",
            "body_x": "0.615",
            "body_y": "0.608",
            "body_z": "0.848",
            "srp_area": "0.52",
            "reflection": "0.5",
            "eccentricity": "0.0009024",
            "inclination": "15.0",
            "raan": "0.0",
            "perigee": "0.0",
        }
        for name, value in defaults.items():
            self.spacecraft_values[name].set(value)

    def _open_spacecraft_settings(self) -> None:
        """Open or focus the modal spacecraft and starting-orbit editor."""
        if (
            self._spacecraft_window is not None
            and self._spacecraft_window.winfo_exists()
        ):
            self._spacecraft_window.lift()
            return

        window = tk.Toplevel(self)
        self._spacecraft_window = window
        window.title("Spacecraft and starting orbit")
        window.geometry("600x660")
        window.resizable(False, False)
        window.transient(self)
        window.configure(background=self.BACKGROUND)

        frame = ttk.Frame(window, style="Panel.TFrame", padding=(28, 18))
        frame.pack(fill="both", expand=True, padx=18, pady=18)
        frame.columnconfigure(1, weight=1)
        ttk.Label(
            frame,
            text="Spacecraft and starting orbit",
            style="PanelTitle.TLabel",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 5))
        ttk.Label(
            frame,
            text="VELOX-C1 values are supplied as defaults. Changes apply only "
            "to the next requested simulation and receive separate cache keys.",
            style="Help.TLabel",
            wraplength=500,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 16))

        fields = (
            ("mass", "Mass", "kg"),
            ("drag_area", "Aerodynamic reference area", "m²"),
            ("body_x", "Body X dimension", "m"),
            ("body_y", "Body Y dimension", "m"),
            ("body_z", "Body Z dimension", "m"),
            ("srp_area", "Solar-radiation area", "m²"),
            ("reflection", "Solar reflection factor", "0–1"),
            ("eccentricity", "Initial eccentricity", "0–1"),
            ("inclination", "Inclination", "deg"),
            ("raan", "RAAN", "deg"),
            ("perigee", "Argument of perigee", "deg"),
        )
        for row, (name, label, unit) in enumerate(fields, start=2):
            ttk.Label(frame, text=label, style="Field.TLabel").grid(
                row=row, column=0, sticky="w", padx=(0, 14), pady=3
            )
            ttk.Entry(
                frame,
                textvariable=self.spacecraft_values[name],
                width=18,
                style="Input.TEntry",
            ).grid(row=row, column=1, sticky="ew", pady=3)
            ttk.Label(frame, text=unit, style="Help.TLabel").grid(
                row=row, column=2, sticky="w", padx=(9, 0), pady=3
            )

        buttons = ttk.Frame(frame, style="Card.TFrame")
        buttons.grid(
            row=len(fields) + 2,
            column=0,
            columnspan=3,
            sticky="e",
            pady=(14, 0),
        )
        ttk.Button(
            buttons,
            text="Reset VELOX-C1",
            style="Secondary.TButton",
            command=self._reset_spacecraft_defaults,
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            buttons,
            text="Done",
            style="Primary.TButton",
            command=window.destroy,
        ).pack(side="left")
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.grab_set()

    def _show_graph_preview(self, output_path: Path) -> None:
        """Select a generated PNG and render it in the preview panel."""
        self._preview_source = output_path
        self.update_idletasks()
        self._render_graph_preview()

    def _schedule_preview_resize(
        self,
        _event: tk.Event[tk.Misc] | None = None,
    ) -> None:
        """Debounce preview resampling while the application is resized."""
        if self._preview_source is None:
            return
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(90, self._render_graph_preview)

    def _render_graph_preview(self) -> None:
        """Fit the source PNG to the preview panel with Lanczos resampling."""
        self._preview_resize_job = None
        if self._preview_source is None:
            return
        target_width = max(self.preview_label.winfo_width() - 4, 100)
        target_height = max(self.preview_label.winfo_height() - 4, 100)
        try:
            with Image.open(self._preview_source) as source:
                source.thumbnail(
                    (target_width, target_height),
                    Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )
                image = ImageTk.PhotoImage(source.copy(), master=self)
        except (OSError, ValueError, tk.TclError) as error:
            self._preview_image = None
            self.preview_label.configure(
                image="", text=f"PNG saved, but preview could not be loaded:\n{error}"
            )
            return
        self._preview_image = image
        self.preview_label.configure(image=image, text="")

    def _start_run(self) -> None:
        """Validate GUI inputs and start the simulation worker thread."""
        try:
            spacecraft = self._parse_spacecraft_parameters()
            request = SimulationRequest.from_bounds(
                launch_date=self._selected_date_text(),
                lower_altitude=self.lower_altitude.get(),
                upper_altitude=self.upper_altitude.get(),
                plot_kind=self.plot_kind.get(),
                use_cache=self.use_cache.get(),
                quick_demo=self.quick_demo.get(),
                spacecraft=spacecraft,
            )
        except ValueError as error:
            messagebox.showerror("Invalid simulation settings", str(error), parent=self)
            return

        self._last_result = None
        self.progress_value.set(0.0)
        self.percent_label.configure(text="0%")
        self.status.set("Starting simulation")
        self._preview_image = None
        self._preview_source = None
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
            self._preview_resize_job = None
        self.preview_label.configure(image="", text="Simulation in progress…")
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")

        worker = threading.Thread(
            target=self._run_in_background,
            args=(request,),
            name="orbital-decay-worker",
            daemon=True,
        )
        worker.start()
        self.after(100, self._poll_events)

    def _run_in_background(self, request: SimulationRequest) -> None:
        """Execute one request without blocking Tk's event loop."""
        try:
            result = run_request(request, OUTPUT_DIRECTORY, self._queue_progress)
        except Exception as error:  # surfaced verbatim in the GUI
            self._events.put(("error", error))
        else:
            self._events.put(("complete", result))

    def _queue_progress(self, percentage: float, message: str) -> None:
        """Transfer a worker progress event to the GUI thread."""
        self._events.put(("progress", (percentage, message)))

    def _poll_events(self) -> None:
        """Apply queued progress, completion, and error events to the GUI."""
        keep_polling = True
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "progress":
                    percentage, message = payload  # type: ignore[misc]
                    self.progress_value.set(percentage)
                    self.percent_label.configure(text=f"{percentage:.0f}%")
                    self.status.set(message)
                elif event == "complete":
                    result = payload
                    assert isinstance(result, RunResult)
                    self._last_result = result
                    self.progress_value.set(100.0)
                    self.percent_label.configure(text="100%")
                    self.status.set(result.message)
                    self._show_graph_preview(result.output_path)
                    self.run_button.configure(state="normal")
                    self.open_button.configure(state="normal")
                    keep_polling = False
                elif event == "error":
                    self.run_button.configure(state="normal")
                    self.status.set("Simulation stopped")
                    self._preview_source = None
                    self.preview_label.configure(
                        image="", text="Graph generation failed. Check the error message."
                    )
                    messagebox.showerror(
                        "Simulation error",
                        f"The graph could not be generated:\n\n{payload}",
                        parent=self,
                    )
                    keep_polling = False
        except queue.Empty:
            pass

        if keep_polling:
            self.after(100, self._poll_events)

    def _open_graph(self) -> None:
        """Open the generated graph with the operating system's viewer."""
        if self._last_result is not None:
            webbrowser.open(self._last_result.output_path.resolve().as_uri())


def main() -> None:
    """Launch the desktop application."""
    app = OrbitalDecayApp()
    app.mainloop()


if __name__ == "__main__":
    main()
