# Contributing

Contributions should preserve physical units, numerical reproducibility, and
clear separation between configuration, propagation, plotting, and interface
code.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before submitting a change

1. Explain the physical or numerical basis for changes to `model.py`.
2. State the source and uncertainty of every added empirical constant.
3. Keep SI units internally and include units in variable names or comments.
4. Add focused tests for configuration or workflow behavior.
5. Run the test suite:

   ```bash
   PYTHONPATH=src python -m unittest discover -s tests -v
   ```

6. Confirm that generated graphs and cache files are absent from the commit,
   except for the labelled website demonstration image.

Scientific changes should include a convergence or validation note when they
affect time steps, orbit sampling, uncertainty propagation, atmospheric
drivers, or force calculations.
