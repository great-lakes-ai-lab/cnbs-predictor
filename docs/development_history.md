# **Forecast Tool Documentation**

## Development History

The forecast tool was developed to improve prediction of Great Lakes hydrologic conditions, with a particular focus on Net Basin Supply (NBS) and its driving components: precipitation (P), evaporation (E), and runoff (R).

### Version 1.0 "MVP (skateboard)"

Initial development focused on building the initial tool framework and training machine learning models to predict NBS directly. This approach allowed us to explore computing resources, python environment, methods, etc.

**a) Data source**

The Climate Forecast System (CFS) and its reanalysis product (CFSR) were selected based on their availability, accessibility, operational nature, and suitable forecast horizon. While alternative datasets, such as ERA5 and the European model, were considered, they may involve costs for data access. Additionally, ERA5 is known to exhibit biases in the Great Lakes region, which could complicate the training and performance of machine learning models.

**b) wgrib2 vs pygrib vs cfgrib**

CFS forecast files are downloaded in their native format: grib2. These files require extra sources/libraries to open, read, and manipulate the data.

While **wgrib2** was developed by NCEP and is an open-source command-lind utility for processing grib2 files. Ultimately, **wgrib2** can easily convert files into netCDF files for easier handling. However, it is an outside library that is not used within python scripts. This would require the NBS-Predictor user to download, make, build the utility prior to using the NBS tool, bringing about its own constraints and issues.

**pygrib** is a library for reading grib files within python. However, it requires installing ECCODES in order to use. In order to do this, the user would need admin access to install or access to a virtual machine.

That left us with **cfgrib**. **cfgrib** actually excels at loading grib files as xarray datasets, best for processing gridded data.

**c) Identifying the domain and creating a mask for each basin (land, lake, and basin)**

Initially, basin masks were generated using shapefiles, and variables were aggregated as total monthly values (e.g., total evaporation and precipitation). However, further analysis revealed that this approach significantly underestimated values due to the coarse representation of basin boundaries within the model grid.

To address this, a grid-aligned masking approach was adopted. By applying masks consistent with the model resolution and calculating spatial means instead of totals, more representative monthly values were obtained.

In earlier iterations, total basin values were also included as model inputs. Since basin values inherently represent the combination of lake and land components, they did not provide additional independent information. As a result, basin values were removed from the input feature set to reduce redundancy.

**d) Framework**

During early development, a stepwise modeling framework was implemented. In this approach, monthly inputs of precipitation (P), runoff (R), and air temperature (AT) are used to predict net basin supply (NBS) on a month-by-month basis. Consequently, a 9-month input sequence produces a corresponding 9-month forecast.

**e) Machine Learning Models**

For simplicity, a **Gaussian Process** approach was used as the only machine learning model.

### Version 1.2 "NBS-Meteo (bike)"

**a) Targets and Data sources**

At this stage, the targets were expanded from NBS to the components, P, E, R. For training, we used data from the **Large Lake Statistical Water Balance Model (L2SWBM)** which has data from 1950 - 2022. The data available on DeepBlue has some issues and Drew Gronewold told us not to use. We used the data published on Zenodo (https://zenodo.org/records/13883098). NBS values were calculated in a post-processing step using the water balance model: P + R - E = NBS.

**b) Machine Learning Models**

The machine learning models section was expanded to include a **Gaussian Process Regressor**, **Linear Regression**, **Random Forest Regressor**, and **Neural Network**. By training more than one ML model, we were able to expand the output into a larger ensemble and skill metrics of methods could be compared.

### Version 1.4 "NBS-Hydro (motorcycle)"

**a) Targets**

In the previous approach, NBS was derived from its individual components, which led to the accumulation of errors and resulted in poor model skill. To address this, NBS was reintroduced as a direct prediction target, allowing the model to implicitly learn and correct for these discrepancies. The model now produces predictions for **P, E, R, and NBS**.

**b) Framework**

At this stage, the modeling framework was transitioned from a stepwise approach to a more streamlined structure. Rather than generating predictions sequentially, the model now ingests the full 9-month forecast as input and produces a complete 12-month forecast in a single pass.

**c) Machine Learning Models**

Under the updated framework, **Linear Regression** no longer provided adequate performance. It was therefore replaced with **XGBoost**, which offered improved predictive capability and better captured nonlinear relationships in the data.

### Version 1.6 "NBS-Predictor (Car)"

This is the **final operational** version to be handed off to the US Army Corps of Engineers.

**a) Anomalies**

The latest iteration of the model transitions from predicting absolute values to predicting anomalies relative to climatology. This change was motivated by slightly improved skill during validation, as anomaly-based targets reduce the influence of strong seasonal cycles and allow the model to focus on deviations that are more directly tied to underlying physical drivers. By removing the dominant climatological signal, the model is better able to learn meaningful patterns and relationships, resulting in modest but consistent gains in predictive performance across evaluation metrics. 

## 2) Major Dataset Decisions

**a) Climate Forecast System (CFS)**

The model development process required a dataset that is operationally reliable, temporally consistent, and suitable for both training and real-time forecasting. We selected the **Climate Forecast System (CFS)** as the primary data source. CFS is a NOAA-operated, fully coupled climate model that provides both a reanalysis product (CFSR/CFSv2) for historical training and real-time forecasts for operational use. This alignment between retrospective and forecast datasets ensures consistency in model inputs across training and deployment. Additionally, CFS produces forecasts four times daily, enabling near real-time updates, and extends predictions out to approximately nine months, which is well-suited for seasonal forecasting applications. The availability of monthly aggregated files further simplifies data handling by reducing file volume and processing overhead.

**b) Alternative Datasets**

- European Centre for Medium-Range Weather Forecasts (ECMWF)

Alternative datasets were evaluated but ultimately not selected. The **ERA5** reanalysis, produced by the **European Centre for Medium-Range Weather Forecasts (ECMWF)**, was evaluated as an alternative dataset due to its high spatial resolution and widespread use. However, several known limitations reduce its suitability for Great Lakes applications. ECMWF technical documentation explicitly notes issues with the representation of large lakes, including “erroneous temperatures of the Great Lakes” [1]. These deficiencies are associated with limitations in lake parameterization and can propagate into biases in near-surface temperature, evaporation, and energy fluxes.

In addition, ERA5 exhibits temporal inconsistencies in earlier portions of the record, including known temperature biases prior to approximately 1967 due to sparse observational constraints [1]. More broadly, studies evaluating ERA5 lake modeling have identified challenges in accurately representing lake thermal structure, including biases in surface temperature seasonality and vertical mixing processes [2]. These limitations introduce additional uncertainty and preprocessing complexity when applying ERA5 to Great Lakes hydrological modeling.


The ECMWF operational forecast system (e.g., Integrated Forecasting System) was also considered due to its strong global performance. However, concerns around long-term data accessibility, licensing restrictions, and the potential for future costs associated with forecast retrieval made it less suitable for a sustainable, operational pipeline.

**References**

[1] European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5: Known Issues. Available at: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
(see “Known Issues” section)

[2] Frontiers in Environmental Science. Evaluation of lake surface temperature and mixing in ERA5 (FLake model assessment). Available at: https://www.frontiersin.org/articles/10.3389/fenvs.2020.609254/full

Overall, **CFS** was selected as the best balance between accessibility, forecast horizon, temporal frequency, and consistency between historical and operational datasets.

## 3) Approaches Tried but Not Adopted

Several alternative approaches were evaluated but ultimately not adopted due to performance or practical constraints. Additional details and results can be found in [Experiments](experiments.md) and [Skill Metrics](experiments_skill_metrics.md).
