# Design Notes

Rationale behind key design choices — the "why" notes that don't belong in
docstrings. Much of this is the development history of the tool: how it evolved
from an MVP to the operational handoff version, and the major decisions made
along the way.

## Development history

The forecast tool was developed to improve prediction of Great Lakes hydrologic
conditions, focused on Net Basin Supply (NBS) and its driving components:
precipitation (P), evaporation (E), and runoff (R). Development proceeded through
a series of versions, framed as a "skateboard → bike → motorcycle → car"
progression toward an operational tool.

### Version 1.0 — "MVP (skateboard)"

Initial framework and first ML models, predicting NBS directly. This stage was
about establishing computing resources, the Python environment, and methods.

- **Data source.** The Climate Forecast System (CFS) and its reanalysis product
  (CFSR) were selected for availability, accessibility, operational nature, and
  forecast horizon. ERA5 and the European model were considered but may involve
  data-access costs, and ERA5 has known biases in the Great Lakes region that
  could complicate ML training. (See "Major dataset decisions" below.)
- **grib2 tooling — wgrib2 vs pygrib vs cfgrib.** CFS files arrive in native
  grib2 and need extra libraries to read.
  - *wgrib2* (NCEP) is a capable command-line utility that can convert to
    netCDF, but it's external to Python and would require users to download,
    make, and build it first.
  - *pygrib* reads grib within Python but requires installing ECCODES, which
    needs admin access or a VM.
  - *cfgrib* was chosen: it loads grib files as xarray datasets, which is best
    for processing gridded data.
- **Domain and basin masks.** Basin masks were first generated from shapefiles
  with variables aggregated as total monthly values, but this underestimated
  values due to coarse basin boundaries on the model grid. A grid-aligned
  masking approach (masks at model resolution, spatial **means** rather than
  totals) gave more representative values. Total basin values were also dropped
  from the inputs — since a basin combines its lake and land components, it added
  redundancy rather than independent information.
- **Framework.** A stepwise model: monthly P, R, and air temperature (AT) →
  predict NBS month by month, so a 9-month input sequence yields a 9-month
  forecast.
- **ML model.** A single **Gaussian Process** model, for simplicity.

### Version 1.2 — "NBS-Meteo (bike)"

- **Targets and data sources.** Targets expanded from NBS to the components
  P, E, R. Training used the **Large Lake Statistical Water Balance Model
  (L2SWBM)**, 1950–2022, from the
  [Zenodo release](https://zenodo.org/records/13883098) (the DeepBlue copy had
  issues and was advised against). NBS was computed in post-processing from the
  water balance: **P + R − E = NBS**.
- **ML models.** Expanded to a **Gaussian Process Regressor**, **Linear
  Regression**, **Random Forest Regressor**, and **Neural Network** — letting the
  outputs form a larger ensemble and enabling skill comparison across methods.

### Version 1.4 — "NBS-Hydro (motorcycle)"

- **Targets.** Deriving NBS from its components accumulated error and produced
  poor skill, so **NBS was reintroduced as a direct prediction target**,
  letting the model implicitly correct for these discrepancies. The model now
  predicts **P, E, R, and NBS**.
- **Framework.** Moved from stepwise to a streamlined single-pass structure: the
  model ingests the full 9-month forecast and produces a complete 12-month
  forecast in one pass.
- **ML models.** Linear Regression was no longer adequate and was replaced with
  **XGBoost**, which better captured nonlinear relationships.

### Version 1.6 — "NBS-Predictor (car)"

The **final operational** version handed off to the US Army Corps of Engineers.

- **Anomalies.** The model transitioned from predicting absolute values to
  predicting **anomalies relative to climatology**. Anomaly-based targets reduce
  the influence of strong seasonal cycles, letting the model focus on deviations
  tied to physical drivers. Removing the dominant climatological signal yielded
  modest but consistent gains across evaluation metrics.

## Major dataset decisions

### Climate Forecast System (CFS) — selected

Development needed a dataset that is operationally reliable, temporally
consistent, and suitable for both training and real-time forecasting. **CFS** —
a NOAA-operated, fully coupled climate model — provides a reanalysis product
(CFSR/CFSv2) for historical training **and** real-time forecasts for operations,
ensuring consistency between training and deployment inputs. It produces
forecasts four times daily (near real-time updates), extends ~nine months (well
suited to seasonal forecasting), and offers monthly aggregated files that
simplify data handling.

### Alternative datasets — evaluated, not selected

- **ERA5 (ECMWF reanalysis).** High spatial resolution and widely used, but
  ECMWF documentation explicitly notes "erroneous temperatures of the Great
  Lakes" tied to lake-parameterization limits, which propagate into near-surface
  temperature, evaporation, and energy-flux biases. ERA5 also has temporal
  inconsistencies early in the record (temperature biases prior to ~1967 from
  sparse observations), and studies note challenges representing lake thermal
  structure (surface-temperature seasonality, vertical mixing). These add
  uncertainty and preprocessing complexity for Great Lakes hydrology.
- **ECMWF operational forecast (e.g. IFS).** Strong global performance, but
  concerns around long-term data accessibility, licensing restrictions, and
  potential future retrieval costs made it less suitable for a sustainable
  operational pipeline.

Overall, **CFS** was the best balance of accessibility, forecast horizon,
temporal frequency, and consistency between historical and operational datasets.

**References**

1. ECMWF, *ERA5: Known Issues* —
   <https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5>
   (see the "Known Issues" section).
2. *Evaluation of lake surface temperature and mixing in ERA5 (FLake model
   assessment)*, Frontiers in Environmental Science —
   <https://www.frontiersin.org/articles/10.3389/fenvs.2020.609254/full>.

## Approaches tried but not adopted

Several alternative approaches were evaluated but not adopted due to performance
or practical constraints. Details and results are recorded on the Experiments
and Skill Metrics pages.

## Models in the operational tool

The operational ensemble combines multiple model families, identified by short
codes in the code and outputs:

- **GP** — Gaussian Process
- **RF** — Random Forest
- **NN** — Neural Network
- **XGB** — XGBoost

Models are trained offline and serialized to `.joblib`; inference loads them via
`joblib.load` and calls `.predict` (see `src.data_processor.CNBSForecaster`).
The operational version predicts **anomalies** relative to climatology
(`mode="anom"`), which are converted back to absolute values in post-processing.

## Smoke checks vs. skill validation

There are two distinct kinds of "is the output OK?" checking, kept separate on
purpose:

- **Smoke checks** (output validators): lightweight, fast, structural sanity
  checks — correct columns, no NaNs/fill values, values within plausible
  physical ranges. They catch catastrophic regressions (sign flips, unit
  errors, schema drift) at the boundary before output is written. They do **not**
  assess forecast quality.
- **Skill validation**: rigorous evaluation of forecast *quality* (e.g.
  RMSE/R²/bias against an archive). See the Skill Metrics page.

The word "validate" is reserved for skill validation; the structural checks use
"smoke check" to avoid collision.
