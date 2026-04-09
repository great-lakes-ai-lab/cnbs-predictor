# Lindsay Fitzpatrick  
**Updated:** 03/04/2026  

## Experiment Summary

| Branch Name | Domain | Initial Conditions | Framework | Variables |
|------------|--------|--------------------|-----------|----------|
| Base       | GL     | None               | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Seasonal   | GL     | Month              | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| SST        | GL     | SST                | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| SWE        | GL     | SWE                | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Chain      | GL     | Month              | 2-step    | Precipitation, Calc. Evaporation, Air Temperature |
| Local      | Lake   | None               | 1-step    | Precipitation, Calc. Evaporation, Air Temperature |

---

## Definitions

### Branch Name
Experiment scripts and full skill metrics can be found under the listed branch name.

### Domain
- **GL (Great Lakes):**  
  All lake variables are used as inputs for all targets.

- **Lake:**  
  Only the corresponding lake variables are used for individual lake targets.  
  Example:  
  `erie_lake_precipitation`, `erie_lake_evaporation`, `erie_lake_air_temperature`  
  → `erie_precipitation`, `erie_evaporation`, `erie_runoff`, `erie_nbs`

### Initial Conditions
- **Month:**  
  Forecast month used as a dummy variable to test improvements to seasonality.

- **SST:**  
  Mean sea surface temperature on the first day of the forecast month for all lakes, used to test improvements to evaporation.

- **SWE:**  
  Mean snow water equivalent on land on the first day of the forecast month.

### Framework
- **1-step:**  
  `precipitation, evaporation, air temperature → precipitation, evaporation, runoff, nbs`

- **2-step:**  
  `precipitation, evaporation, air temperature → precipitation, evaporation, runoff → nbs`