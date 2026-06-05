# Experiments

A summary of the experiment configurations explored during development. Each
experiment varies the domain, initial conditions, framework, or input variables
to test a specific hypothesis about what improves forecast skill. Results for
these experiments are recorded on the Skill Metrics page.

_Last updated from source: 03/04/2026._

## Experiment summary

| Branch Name | Domain | Initial Conditions | Framework | Variables |
|-------------|--------|--------------------|-----------|-----------|
| Base        | GL     | None               | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Seasonal    | GL     | Month              | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| SST         | GL     | SST                | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| SWE         | GL     | SWE                | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Chain       | GL     | Month              | 2-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Local       | Lake   | None               | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |

## Definitions

### Branch name

Experiment scripts and full skill metrics can be found under the listed branch
name.

### Domain

- **GL (Great Lakes):** all lake variables are used as inputs for all targets.
- **Lake:** only the corresponding lake's variables are used for that lake's
  targets. For example:
  `erie_lake_precipitation`, `erie_lake_evaporation`, `erie_lake_air_temperature`
  → `erie_precipitation`, `erie_evaporation`, `erie_runoff`, `erie_nbs`.

### Initial conditions

- **Month:** forecast month used as a dummy variable, to test improvements to
  seasonality.
- **SST:** mean sea surface temperature on the first day of the forecast month,
  for all lakes — to test improvements to evaporation.
- **SWE:** mean snow water equivalent on land on the first day of the forecast
  month.

### Framework

- **1-step:**
  `precipitation, evaporation, air temperature → precipitation, evaporation, runoff, nbs`
- **2-step:**
  `precipitation, evaporation, air temperature → precipitation, evaporation, runoff → nbs`
