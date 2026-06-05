# Usage

Most workflows run through the Jupyter notebooks under `notebooks/production/`.
The `src/` package provides the underlying library (data download, processing,
forecasting, and output checks) documented in the {doc}`api/index`.

## Running Jupyter Lab

After setting up and activating the `nbs_env` environment:

```bash
jupyter lab
```

## Working with the notebooks

1. In Jupyter Lab, navigate to `notebooks/production/`.
2. Open the appropriate notebook. There are separate notebooks for:
   - **Training** a forecast model (not needed for most users),
   - **Downloading and preprocessing** input data from NOAA CFS and other sources,
   - **Generating forecasts** (e.g. `2_LEF_forecast_model.ipynb`).
3. Set your directory paths in the **User Input** section near the top.
4. Run the notebook to generate forecasts.

## Inputs and data sources

Forecasts require NOAA CFS atmospheric forecast data and GLSEA sea-surface
temperatures as initial conditions. The download/preprocessing notebooks handle
acquiring and shaping both. See the README "Data Sources" table for the full
list of datasets and how they are used (forecasting vs. training).

## Running the tests

The library has an automated test suite. To run the fast, offline tests:

```bash
pytest -m "not network"
```

Tests that reach live NOAA/AWS endpoints are marked `network` and can be run
explicitly with `pytest -m network`. On pull requests, continuous integration
runs the offline suite automatically. See `CONTRIBUTING.md` for details.
