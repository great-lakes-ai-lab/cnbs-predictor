# Design Notes

Rationale behind key design choices. These are the "why" notes that don't
belong in docstrings. Expand as decisions are made.

## Forecasting NBS directly

NBS is forecast **directly**, rather than being derived by combining separate
precipitation, evaporation, and runoff forecasts. Per the project README, this
"reduce[s] the accumulation of error and improve[s] overall forecast
reliability" — errors in three separately-modeled components would otherwise
compound when summed into NBS.

The pipeline still forecasts the individual components (P, E, R) as well, but
NBS is its own target.

## Models

Forecasts are produced by more than one model and combined. The current models
loaded at inference are identified by short codes (e.g. `GP`, `RF`):

- **GP** — Gaussian Process
- **RF** — Random Forest

Models are trained offline and serialized to `.joblib`; inference loads them via
`joblib.load` and calls `.predict` (see `src.data_processor.CNBSForecaster`).
TensorFlow is **not** required for inference — it appears only in the training
workflow.

> _To expand: why these model families, how predictions are combined/averaged,
> and how anomalies vs. absolute values are handled (`mode="anom"`)._

## Smoke checks vs. skill validation

There are two distinct kinds of "is the output OK?" checking, kept separate on
purpose:

- **Smoke checks** (output validators): lightweight, fast, structural sanity
  checks — correct columns, no NaNs/fill values, values within plausible
  physical ranges. They catch catastrophic regressions (sign flips, unit
  errors, schema drift) at the boundary before output is written. They do **not**
  assess forecast quality.
- **Skill validation**: the rigorous evaluation of forecast *quality* (e.g.
  RMSE/CRPS/R² against an archive). This is a separate workstream.

The word "validate" is reserved for skill validation; the structural checks use
"smoke check" to avoid collision.

## Inputs and data sources

Forecasts require NOAA CFS atmospheric forecast data plus GLSEA sea-surface
temperatures as initial conditions. Training additionally uses CFSR, L2SWBM, and
GLCC datasets. See the README "Data Sources" table for the authoritative list.

> _To expand: data layout expectations, the GLCC sentinel fill values
> (-99990.0 / -9999.0) and where they come from, and the cfs forecast database
> schema._
