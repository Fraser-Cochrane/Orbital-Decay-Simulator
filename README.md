# VELOX-C1 Orbital Decay Model

A research-oriented Monte Carlo model for estimating VELOX-C1 orbital decay
between 500 and 700 km. The package combines NRLMSIS 2.1 atmospheric data,
composition-dependent Sentman/DRIA drag, vector solar-radiation pressure,
first-order J2 secular effects, and propagated environmental uncertainty.

The project provides an interactive FastAPI website, a Tkinter desktop
interface, command-line workflows, high-resolution Matplotlib figures, and
exact-match result caching.

## Scientific capabilities

- NRLMSIS 2.1 density, neutral composition, and temperature evaluated using
  epoch and geodetic position.
- Solar F10.7, centred 81-day F10.7, and storm-time Ap atmospheric drivers.
- Composition- and temperature-dependent Sentman/DRIA free-molecular drag.
- Atomic-oxygen adsorption and incomplete gas-surface accommodation.
- First-order J2 secular evolution of the ascending node and perigee.
- Sun-directed vector solar-radiation pressure with inverse-square scaling and
  finite-disc Earth eclipse geometry.
- Monte Carlo propagation of atmospheric, space-weather, wind, and
  gas-surface-interaction uncertainty.
- Data-estimated F10.7/Ap innovation dependence for forecast intervals.
- Exact-match caching keyed by code, inputs, driver data, and numerical setup.

The scientific profile uses 200 realizations and a three-hour propagation
step. The optional quick-demo profile uses eight realizations and a 30-day
step so the complete interface can be demonstrated rapidly. Quick-demo output
is explicitly labelled and is not suitable for scientific interpretation.

## Installation

Python 3.10 or newer is required. Tk support is needed only for the desktop
interface.

```bash
git clone <repository-url>
cd velox-c1-orbital-decay
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python run_web.py
```

Open `http://127.0.0.1:8000` to use the simulator in a browser. To launch the
desktop interface instead, run:

```bash
python run_gui.py
```

On macOS, `start_web.command` provides a one-click alternative: double-click
it in Finder, allow the first-run environment setup to finish, and leave the
visible Terminal window open while using the website. The launcher waits for
the service to become healthy before opening the browser. If a previously
opened page reports that the service is unavailable, launch it again and use
the page's **Try again** button.

JupyterLab users can open `launch_web.ipynb` from the repository directory and
run its launch cell. The notebook starts a fresh local service on an available
port, waits for the health check, and provides a direct link to the simulator.

For a notebook stored in a separate Jupyter directory, copy
`open_website.ipynb` instead. Its single `PROJECT_FOLDER` setting points to the
complete source package, including the required `docs/` website assets.
`master.ipynb` provides the same separate-folder launch flow under an explicit
master-launcher name and includes a stop-server cell. Each execution starts a
fresh server on an available local port, ensuring that the website loads the
current source files instead of reconnecting to an older process on port 8000.

On Windows, activate the environment with `.venv\Scripts\activate`. Tkinter
normally ships with Python; some Linux distributions provide it separately as
`python3-tk`.

## Desktop interface

The interface supports two outputs:

- **Line graph:** one-year mean semi-major-axis decay rate across 20 evenly
  spaced initial altitudes.
- **Heat map:** instantaneous mean decay rate across altitude and percentage
  phase through a solar-cycle ascent interval.

Launch dates use separate year, month, and day selectors. The selectors stop
at the latest date for which the installed PyMSIS space-weather file covers the
complete calculation. Lower and upper altitude controls are restricted to the
500–700 km model domain. Spacecraft mass, geometry, optical area, reflection
factor, and starting orbital elements can be edited from the advanced panel.

Every generated graph is exported at 600 DPI and previewed in the application
with high-quality Lanczos resampling. Output files and cache data are written
below `outputs/` and excluded from Git commits.

Line graphs retain each simulated altitude as a visible point. Matplotlib
renders a weighted exponential regression of the mean, while a shape-preserving
curve displays the relative one-standard-deviation envelope. The regression is
display-only and does not create or modify simulation values.

## Command line

```bash
python run_cli.py --start 2024-01-01 --no-show
python run_cli.py --start 2022-01-01 --start 2023-01-01 --workers 2 --no-show
python run_cli.py --solar-cycle-comparison --workers 3 --no-show
python run_cli.py --solar-cycle-surface --no-show
```

An editable installation also provides the `velox-decay`, `velox-decay-gui`,
and `velox-decay-web` commands.

## Interactive website

The responsive browser client is located in `docs/` and is served by the
FastAPI application in `velox_decay.web`. It provides:

- line-graph and heat-map selection;
- server-derived launch-date limits;
- lower and upper altitude fields with 20 inclusive samples;
- editable VELOX-C1 spacecraft values;
- quick-demo and exact-cache controls;
- live queued and numerical progress; and
- direct in-page graph preview and PNG download.

The page must be opened through the Python application, not as a `file://` URL.
GitHub Pages can show static files but cannot execute PyMSIS or the Monte Carlo
propagator. Use a Python application host or the included container for the
working simulator.

The service deliberately runs one numerical job at a time. The calculation
workflows use temporary process-wide model settings, so one worker preserves
isolation while additional requests wait in a bounded queue. For an
internet-facing deployment, add platform-level rate limiting or authentication
appropriate to the expected audience.

See [DEPLOYMENT.md](DEPLOYMENT.md) for publication steps and URL placeholders
that must be replaced for a specific domain.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The unit suite covers configuration validation, date coverage, altitude
sampling, cache isolation, spacecraft profiles, workflow behavior, and export
resolution. Full scientific propagation is intentionally excluded from the
fast test suite because it is computationally expensive.

GitHub Actions runs the test suite and builds the Python distribution on every
push and pull request.

## Repository layout

```text
.
├── .github/workflows/tests.yml
├── docs/
│   ├── app.js
│   ├── assets/decay-preview.png
│   ├── index.html
│   └── styles.css
├── outputs/
├── src/velox_decay/
│   ├── cli.py
│   ├── config.py
│   ├── gui.py
│   ├── model.py
│   ├── plotting.py
│   ├── web.py
│   └── workflows.py
├── tests/
├── CONTRIBUTING.md
├── DEPLOYMENT.md
├── pyproject.toml
├── launch_web.ipynb
├── master.ipynb
├── open_website.ipynb
├── run_cli.py
├── run_gui.py
├── run_web.py
└── start_web.command
```

## Scientific scope

This software is intended for research and educational analysis. It is not a
flight-dynamics, navigation, operational re-entry, or conjunction-assessment
service. Reported uncertainty covers only the implemented stochastic inputs
and model-discrepancy terms; unknown attitude behavior and unmodelled
environmental effects may introduce additional error.

## License

No open-source license is included. Under standard copyright rules, that means
reuse rights are not granted automatically. Add an appropriate `LICENSE` file
before publication if external copying, modification, or redistribution should
be permitted.
