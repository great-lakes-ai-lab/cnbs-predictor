### **Forecast Tool Documentation**

## 1) Development History

The forecast tool was developed to improve prediction of Great Lakes hydrologic conditions, with a particular focus on Net Basin Supply (NBS) and its driving components: precipitation (P), evaporation (E), and runoff (R).

# Version 1.0 "MVP (skateboard)"

Initial development focused on building the initial tool framework and training machine learning models to predict NBS directly. This approach allowed us to explore computing resources, python environment, methods, etc.

a) Data source

The Climate Forecast System (CFS) and its reanalysis product (CFSR) were selected based on their availability, accessibility, operational nature, and suitable forecast horizon. While alternative datasets, such as ERA5 and the European model, were considered, they may involve costs for data access. Additionally, ERA5 is known to exhibit biases in the Great Lakes region, which could complicate the training and performance of machine learning models.

b) wgrib2 vs pygrib vs cfgrib

CFS forecast files are downloaded in their native format: grib2. These files require extra sources/libraries to open, read, and manipulate the data.

While **wgrib2** was developed by NCEP and is an open-source command-lind utility for processing grib2 files. Ultimately, **wgrib2** can easily convert files into netCDF files for easier handling. However, it is an outside library that is not used within python scripts. This would require the NBS-Predictor user to download, make, build the utility prior to using the NBS tool, bringing about its own constraints and issues.

**pygrib** is a library for reading grib files within python. However, it requires installing ECCODES in order to use. In order to do this, the user would need admin access to install or access to a virtual machine.

That left us with **cfgrib**. **cfgrib** actually excels at loading grib files as xarray datasets, best for processing gridded data.

c) Identifying the domain and creating a mask for each basin (land, lake, and basin)

Initially, basin masks were generated using shapefiles, and variables were aggregated as total monthly values (e.g., total evaporation and precipitation). However, further analysis revealed that this approach significantly underestimated values due to the coarse representation of basin boundaries within the model grid.

To address this, a grid-aligned masking approach was adopted. By applying masks consistent with the model resolution and calculating spatial means instead of totals, more representative monthly values were obtained.

In earlier iterations, total basin values were also included as model inputs. Since basin values inherently represent the combination of lake and land components, they did not provide additional independent information. As a result, basin values were removed from the input feature set to reduce redundancy.

d) Framework

During early development, a stepwise modeling framework was implemented. In this approach, monthly inputs of precipitation (P), runoff (R), and air temperature (AT) are used to predict net basin supply (NBS) on a month-by-month basis. Consequently, a 9-month input sequence produces a corresponding 9-month forecast.

e) Machine Learning Models

For simplicity, a **Gaussian Process** approach was used as the only machine learning model.

# Version 1.2 "NBS-Meteo (bike)"

a) Targets and Data sources

At this stage, the targets were expanded from NBS to the components, P, E, R. For training, we used data from the **Large Lake Statistical Water Balance Model (L2SWBM)** which has data from 1950 - 2022. The data available on DeepBlue has some issues and Drew Gronewold told us not to use. We used the data published on Zenodo (https://zenodo.org/records/13883098). NBS values were calculated in a post-processing step using the water balance model: P + R - E = NBS.

b) Machine Learning Models

The machine learning models section was expanded to include a **Gaussian Process Regressor**, **Linear Regression**, **Random Forest Regressor**, and **Neural Network**. By training more than one ML model, we were able to expand the output into a larger ensemble and skill metrics of methods could be compared.

# Version 1.4 "NBS-Hydro (motorcycle)"

a) Targets

In the previous approach, NBS was derived from its individual components, which led to the accumulation of errors and resulted in poor model skill. To address this, NBS was reintroduced as a direct prediction target, allowing the model to implicitly learn and correct for these discrepancies. The model now produces predictions for **P, E, R, and NBS**.

b) Framework

At this stage, the modeling framework was transitioned from a stepwise approach to a more streamlined structure. Rather than generating predictions sequentially, the model now ingests the full 9-month forecast as input and produces a complete 12-month forecast in a single pass.

c) Machine Learning Models

Under the updated framework, **Linear Regression** no longer provided adequate performance. It was therefore replaced with **XGBoost**, which offered improved predictive capability and better captured nonlinear relationships in the data.

# Version 1.6 "NBS-Predictor (Car)"

This is the **final operational** version to be handed off to the US Army Corps of Engineers.


## 2) Major Dataset Decisions

Several key decisions were made regarding data selection and preprocessing:



## 3) Operational Constraints for Inputs

To function in an operational or near-operational setting, the tool must adhere to several constraints:

Data availability

Inputs must be available in near real-time or with minimal latency.

Consistency

Input datasets must be continuously updated and version-stable to avoid forecast discontinuities.

Completeness

Missing data must be handled robustly (gap-filling or fallback strategies).

Latency tolerance

Forecast generation must occur within a defined time window to be useful for decision-making.

Format standardization

Inputs must conform to consistent formats (e.g., NetCDF, structured arrays, or database schema).

Scalability

The system must handle expanding datasets and longer forecast horizons without significant performance degradation.

Model input requirements

Features must be available at forecast time (no reliance on future or unavailable variables).

## 4) Approaches Tried but Not Adopted

Several approaches were explored but ultimately not adopted due to performance or practical limitations:
