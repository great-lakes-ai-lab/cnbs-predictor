### **Forecast Tool Documentation**

## 1) Development History

The forecast tool was developed to improve prediction of Great Lakes hydrologic conditions, with a particular focus on Net Basin Supply (NBS) and its driving components: precipitation (P), evaporation (E), and runoff (R).

# Version 1.0 "MVP (skateboard)"

Initial development focused on building the initial tool framework and training machine learning models to predict NBS directly. This approach allowed us to explore computing resources, python environment, methods, etc.

a) wgrib2 vs pygrib vs cfgrib

CFS forecast files are downloaded in their native format: grib2. These files require extra sources/libraries to open, read, and manipulate the data.

While **wgrib2** was developed by NCEP and is an open-source command-lind utility for processing grib2 files. Ultimately, **wgrib2** can easily convert files into netCDF files for easier handling. However, it is an outside library that is not used within python scripts. This would require the NBS-Predictor user to download, make, build the utility prior to using the NBS tool, bringing about its own constraints and issues.

**pygrib** is a library for reading grib files within python. However, it requires installing ECCODES in order to use. In order to do this, the user would need admin access to install or access to a virtual machine.

That left us with **cfgrib**. **cfgrib** actually excels at loading grib files as xarray datasets, best for processing gridded data.

b) Identifying the domain and creating a mask for each basin (land and lake)



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
