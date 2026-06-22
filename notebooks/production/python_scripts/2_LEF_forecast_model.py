# %% [markdown]
# # Step 2: The Official NBS-Predictor
# Lindsay Fitzpatrick
# ljob@umich.edu
# 
# **Initial Version:** 08/19/2024  
# **Last Updated:** 06/01/2026  
# 
# This script generates **Net Basin Supply (NBS)** forecasts as well as it's components; **precipitation**, **evaporation**, and **runoff**;for each of the Great Lakes using trained machine learning models and processed CFS forecast data. It performs the following key tasks:
# 
# 1. Loads the input SQLite database containing pre-processed CFS forecast data.
# 2. Applies one or more trained models to predict monthly CNBS components.
# 3. Generates an ensemble of CNBS forecasts across all available CFS runs.
# 4. Outputs forecast results in both database and CSV formats.
# 5. Produces both static and interactive visualizations of forecast results.
# 
# ### Required Inputs
# - **Forecast Data:** `cfs_forecast_data.db`
# - **SNODAS Data:** `SWE.SNODAS.GL.csv` 
# - **Scalers:** `x_scaler.joblib`, `y_scaler.joblib`  
# - **Trained Models:** One or more of the following:
#   - `GP_trained_model.joblib`
#   - `RF_trained_model.joblib`
#   - `XGB_trained_model.joblib`
#   - `NN_trained_model.joblib`
# 
# 
# ### Output Files (saved to `data/forecast/`)
# - **`CNBS_forecast.csv`**  
#   Full ensemble forecast results across all CFS runs for each lake and component (precipitation, evaporation, runoff, total NBS).
# 
# - **`CNBS_forecast_formatted_cms.csv`**  
#   A simplified CSV containing the monthly mean forecast for each lake and component. Useful for quick interpretation or public-facing applications.
# 
# - **`cnbs_forecast.db`**  
#   A SQLite database version of the forecast results, equivalent to the CSV outputs, intended for structured data access and downstream integration.
# 
# - **`CNBS_forecasts.png`**  
#   A static 4×4 figure showing CNBS forecasts: each row represents one of the Great Lakes, and each column shows a CNBS component (precip, evap, runoff, NBS).
# 
# - **`CNBS_forecasts_interactive.html`**  
#   An interactive version of the PNG plot, allowing users to hover over lines to explore forecast values for each component and lake in greater detail.
# 
# This script enables transparent, reproducible CNBS forecasting workflows with both technical and user-friendly outputs.

# %%
import pandas as pd
import os
import sys
#import warnings

# Add the path to the src directory (two levels up)
sys.path.append(os.path.abspath('../../'))
from src.data_processor import *
from src.database_utils import CFSDatabase
from src.utilities import get_first_forecast_month
from src.hydro_utils import convert_mm_to_cms
from src.data_loader import DataLoader

# %% [markdown]
# # User Input
# 
# ### Set Up File Paths, Data Sources, and Model Parameters
# 
# This cell establishes the key paths and configuration for the forecasting workflow:
# 
# - **Local and Data Directories:**  
#   - `local_path`: Base directory where the repository is cloned.  
#   - `input_dir`: Path to the `data` folder within the repository.  
# 
# - **CFS Forecast Database:**  
#   - `cfs_database` and `cfs_table` specify the SQLite database and table containing CFS forecast data.  
# 
# - **CNBS Forecast Output Database:**  
#   - `cnbs_database` and `cnbs_table` define where processed CNBS forecasts will be saved.  
# 
# - **Great Lakes Surface Water Temperature Data:**  
#   - `url` points to the NOAA GLERL Coastwatch `.dat` file used to retrieve lake surface temperatures.  
# 
# - **Model and Scaler Directories:**  
#   - `model_dir` and `scaler_dir` indicate the locations of trained machine learning models and scalers used for prediction.  
# 
# - **Models to Run:**  
#   - `models_info` lists the CNBS prediction models that will be executed (`GP`, `RF`, `NN`, `XGB`).

# %%
# Directory where the repository is cloned
local_path = '/Users/ljob/Desktop/'

# Path to data directory
input_dir = local_path + 'cnbs-predictor/data/'

# Path to input CFS forecast database
cfs_database = local_path + 'cfs_forecast_data.db'
cfs_table = 'cfs_forecast_data'

# Path to save CNBS forecast output to a database
cnbs_database = input_dir + 'forecast/cnbs_forecast.db'
cnbs_table = 'cnbs_forecast'

# Path to saved data
model_dir = input_dir + "input/models/"
scaler_dir = input_dir + "input/scalers/"
clim_dir = input_dir + "input/climatology/"

# Path to probability csv files
probability_dir = input_dir + "probabilities/"

# List of models to run
models_info = ['GP', 'RF', 'NN', 'XGB']

# %% [markdown]
# # Begin Script

# %% [markdown]
# ### Load and Prepare CFS Forecast Data for CNBS Prediction
# 
# This cell handles retrieving and formatting the latest CFS forecast data from the database in preparation for CNBS prediction:
# 
# - **Load Data:** Uses the `CFSDatabase` class to load forecast data from the specified database and table.  
# - **Structure Input:** The `CFSProcessor.structure_input()` function formats the raw data into a model-ready feature matrix `X`.  
# - **Check for Missing Values:**  
#   - Verifies if there are any NaNs in `X`.  
#   - Prints out the rows containing missing values for review.  
#   - Drops rows with NaN values to ensure clean input for the prediction models.

# %%
# Load CFS forecast data from the database and format it before predicting CNBS
data = CFSDatabase(cfs_database, cfs_table).load()

# Filter to only include forecast data for the first forecast month and beyond.
# This saves time and resources by not including older/archived data.
data_filtered = CFSTransformer(data).filter()

# %% [markdown]
# ### Load Saved Climatology Processors
# 
# This cell loads previously saved `SeasonalCycleProcessor` objects used for climatology-based anomaly transformations.
# 
# The `load_clim()` function restores both:
# 
# - `scp_X` → climatology processor for input features
# - `scp_y` → climatology processor for target variables
# 
# from a directory structure containing:
# 
# - `inputs/climatology.csv`
# - `inputs/metadata.json`
# - `targets/climatology.csv`
# - `targets/metadata.json`
# 
# These processors store the monthly climatological means and metadata required to:
# 
# - convert absolute values to anomalies
# - reconstruct anomalies back into physical values
# - maintain consistent preprocessing between training and operational forecasting workflows
# 
# Loading saved climatology processors ensures reproducibility and guarantees that forecasts use the same seasonal baseline applied during model training.

# %%
climatology = SeasonalCycleProcessor()
scp_X, scp_y = climatology.load_clim(clim_dir)
scp_y.climatology.columns = [
    c.replace("_target", "") for c in scp_y.climatology.columns
]   

# %% [markdown]
# ### Reshape Forecast Data from Long Format to Lead-Wide Format
# 
# This cell uses the `CFSTransformer` class and `structure_input` function to convert the Climate Forecast System (CFS) forecast dataset from a **long/tabular format** into a **lead-wide machine learning format**.  
# 
# The original dataset from the database stores each forecast variable as a separate row identified by:
# 
# - `cfs_run`
# - `lake`
# - `surface_type`
# - `component`
# - forecast `year` and `month`
# 
# The transformation:
# 
# 1. Computes the forecast lead month (`mo0`, `mo1`, `mo2`, etc.) relative to the initialization date (`cfs_run`)
# 2. Combines metadata columns into descriptive variable names such as:
#    - `superior_lake_precipitation_mo0`
#    - `erie_land_evaporation_mo3`
# 3. Pivots the dataframe so each `cfs_run` becomes a single row
# 4. Produces a wide-format dataframe suitable for:
#    - machine learning model training
#    - forecast verification
#    - feature engineering
#    - ensemble prediction workflows
# 
# The resulting dataframe contains one row per forecast initialization time and one column per lake/component/lead combination.
# 

# %%
X = CFSTransformer(data_filtered).structure_input()

# %% [markdown]
# ### Convert Forecast Inputs to Anomalies
# 
# This cell converts the merged forecast predictor dataset from absolute values into climatological anomalies using the previously loaded input climatology processor (`scp_X`).
# 
# The `subtract_climatology_leadwide()` function:
# 
# 1. Identifies the forecast lead month for each variable (`mo0`, `mo1`, `mo2`, etc.)
# 2. Determines the corresponding verifying calendar month for each forecast lead
# 3. Subtracts the monthly climatological mean from each predictor value
# 4. Produces anomaly-based predictors centered around the historical seasonal cycle
# 
# This transformation removes the mean seasonal signal and allows the machine learning models to focus on departures from normal conditions rather than absolute magnitudes.
# 
# The resulting dataframe (`X_anom`) maintains the same structure and dimensions as the original lead-wide dataframe while representing all predictors in anomaly space.

# %%
X_anom = climatology.subtract_climatology_leadwide(X, scp_X)

# %% [markdown]
# ### Add Temporal Features and Reorganize Predictor Columns
# 
# This cell augments the predictor dataset with temporal features and reorganizes all variables into a consistent feature ordering required for machine learning inference.
# 
# The `add_time_features()` function adds:
# 
# - `month_sin`
# - `month_cos`
# 
# to represent the cyclical nature of the calendar year, preserving seasonal continuity between December and January.
# 
# It also adds:
# 
# - `time`
# 
# which represents a continuous normalized time trend used to capture long-term temporal variability and nonstationarity in the dataset.
# 
# After temporal feature generation, the script defines a strict `feature_column_order` to ensure all predictors appear in the exact order expected by the trained machine learning models.
# 
# The ordered feature set includes:
# 
# 1. Temporal predictors
# 2. Basin-scale SWE predictors
# 3. Basin-scale SST predictors
# 4. Atmospheric forecast predictors for:
#    - precipitation
#    - evaporation
#    - air temperature
# 
# across:
# 
# - all four Great Lakes basins
# - lake and land surface types
# - forecast lead months (`mo0`–`mo9`)
# 
# Finally, the dataframe is reordered to match the training feature structure used during model development, ensuring consistency between operational forecasts and model training workflows.

# %%
X_time = CFSTransformer(X_anom).add_time_features()

feature_column_order = (
    ['time', 'month_sin', 'month_cos'] +
    [
        f'{lake}_{surface_type}_{comp}_mo{m}'
        for lake in ['superior', 'michigan-huron', 'erie', 'ontario']
        for surface_type in ['lake', 'land']
        for comp in ['precipitation', 'evaporation', 'air_temperature']
        for m in range(10) #from 0 to 9, representing the 10 months of lead time
    ]
)

# Reorder the columns to match the specified order
X_reorg = X_time[feature_column_order]

# %% [markdown]
# ### Run CNBS Forecast Predictions
# 
# This cell initializes the `CNBSForecaster` with the directories containing trained models and scalers. It then uses the forecaster to generate predictions for each model in the directory.  
# 
# Steps performed:
# 
# 1. **Initialize Forecaster**  
#    The `CNBSForecaster` loads all trained models and associated scalers (`x_scaler` and `y_scaler`) for transforming input features and reversing predictions.
# 
# 2. **Predict with Each Model**  
#    Iterates through each model (`GP`, `NN`, `RF`, `XGB`, etc.) and generates CNBS forecasts for all lakes and variables. The predictions include 12 months (`mo0` to `mo11`) for precipitation, evaporation, runoff, and NBS.
# 
# 3. **Combine Predictions**  
#    Concatenates the predictions from all models into a single DataFrame `df_all`, adding a `model` column to identify which model produced each set of forecasts.
# 
# **Outputs:**  
# - `df_all` – DataFrame with predicted CNBS values for each lake, component, forecast month (encoded as `mo0`–`mo11`), and model.

# %%
# Initialize model forecaster
model_runner = CNBSForecaster(model_dir, scaler_dir, "anomaly")

# Run predictions for each model
model_predictions = []
for model_name in model_runner.models.keys():
    print(f"Predicting with {model_name}...")
    df_y = model_runner.predict(X_reorg, model_name, scp_y)
    model_predictions.append(df_y)

# Combine all predictions
df_all = pd.concat(model_predictions).reset_index()

#Convert cfs_run back to YYYYMMDDHH format
df_all["cfs_run"] = pd.to_datetime(df_all["cfs_run"]).dt.strftime("%Y%m%d%H")

# %% [markdown]
# ### Formatting and Saving Forecast Output
# 
# This cell reshapes and processes the forecast results to match the structure of the original input data, preparing it for storage and downstream use.
# 
# #### Key Steps:
# 
# - **Reshaping Output**:  
#   The forecast DataFrame is **melted** into a long-format structure, similar to how the original input data is stored in the database (i.e., one row per variable, per time step).
# 
# - **Saving to Database**:  
#   The processed forecast output is saved to a **user-specified database**, preserving compatibility with existing tools and workflows.
# 
# This final step ensures that the forecast results are cleanly formatted, physically interpretable, and readily accessible for operational decision-making or further analysis.
# 

# %% [markdown]
# The cell below generates a formatted CSV similar to a tracking file maintained by USACE.
# 
# ### Example Output:
# 
# | cfs_run    | model | lake          | component    | month_0      | month_1      | month_2      | month_3      | month_4     | month_5     | month_6     | month_7     | month_8     | month_9     | month_10    | month_11    |
# |------------|-------|---------------|-------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
# | 2024090400 | GP    | erie          | evaporation | 169.286182   | 175.525651   | 155.942570   | 105.221193   | 49.588555   | 54.368630   | 30.514956   | 15.451097   | 24.529560   | 57.125326   | 47.897625   | 101.213959  |
# | 2024090400 | GP    | erie          | nbs         | -179.909587  | -30.245478   | 24.279292    | 29.282516    | 91.002201   | 46.111309   | 181.265523  | 257.803210  | 188.925473  | 16.504276   | 65.490739   | -117.337339 |
# | 2024090400 | GP    | erie          | precipitation | 53.653021  | 103.950400   | 112.203127   | 60.024030    | 60.194799   | 41.491717   | 70.158099   | 91.900693   | 82.856855   | 59.733797   | 74.811895   | 41.064274   |
# | 2024090400 | GP    | erie          | runoff       | 9.760347    | 42.087138    | 72.079224    | 66.216000    | 63.664174   | 50.553577   | 73.335749   | 112.666789  | 81.401896   | 26.485872   | 28.968797   | 0.632830    |
# | 2024090400 | GP    | michigan-huron | evaporation | 77.769942   | 87.448243    | 115.609391   | 111.194251   | 71.208255   | 71.408165   | 46.503628   | 19.151783   | 14.076061   | 25.807311   | 7.845980    | 34.349577   |
# 

# %%
transformer = ForecastTransformer(df_all)

df_melt = transformer.melt()

# Optional - Save to a CSV
#df_melt.to_csv(input_dir + 'forecast/CNBS_forecast_months.csv', sep='\t', index=False)

# %% [markdown]
# This cell pivots the forecast DataFrame using the `ForecastTransformer` object to create a tidy, analysis-ready format. 
# 
# - The variables `precipitation`, `evaporation`, `runoff`, and `nbs` are kept as separate columns.  
# - A new column, `forecast_month`, is added, which calculates the actual calendar month each forecast value is valid for based on the `cfs_run` date and the original month offsets (e.g., `precipitation_mo0` → `forecast_month` for month 0).  
# - The resulting `df_pivoted` DataFrame is then saved as a tab-separated CSV file in the `forecast` directory for downstream analysis or plotting.
# 

# %% [markdown]
# ### Example Output:
# 
# | cfs_run    | forecast_month | model | lake           | precipitation | evaporation | runoff      | nbs         |
# |------------|----------------|-------|----------------|---------------|------------|------------|------------|
# | 2024090400 | 2024-09        | GP    | erie           | 53.653021     | 169.286182 | 9.760347   | -179.909587 |
# | 2024090400 | 2024-09        | GP    | michigan-huron | 62.134949     | 77.769942  | 33.456800  | -28.263310  |
# | 2024090400 | 2024-09        | GP    | ontario        | 47.591716     | 88.953374  | 51.670611  | -62.127769  |
# | 2024090400 | 2024-09        | GP    | superior       | 77.634790     | 68.623162  | 38.261790  | 36.415108   |
# | 2024090400 | 2024-10        | GP    | erie           | 103.950400    | 175.525651 | 42.087138  | -30.245478  |
# 

# %%
df = df_all.copy()
df["cfs_run"] = pd.to_datetime(df["cfs_run"], format ="%Y%m%d%H")

# Step 1: Melt into long format
df_melted = df.melt(
    id_vars=['cfs_run', 'model'],
    var_name='variable',
    value_name='value'
)

df_pivoted = transformer.pivot()

# Optional - Save the pivoted dataframe to a CSV file
#df_pivoted.to_csv(input_dir + 'forecast/CNBS_forecast_wide.csv', sep='\t', index=False)

# %% [markdown]
# ### Example Output:
# 
# | cfs_run    | year | month | model | lake          | component     | value [mm]         | value [cms]       |
# |------------|------|-------|-------|---------------|---------------|------------------|-----------------|
# | 2025030300 | 2025 | 3     | GP    | erie          | precipitation | 167.0919492238333 | 1600.4868418971937 |
# | 2025030300 | 2025 | 3     | GP    | michigan-huron| precipitation | 156.39444315148768 | 6850.022890319024 |
# | 2025030300 | 2025 | 3     | GP    | ontario       | precipitation | 136.86474390600262 | 971.3492820001507 |
# | 2025030300 | 2025 | 3     | GP    | superior      | precipitation | 64.49981001019145  | 1977.0164659523177 |
# | 2025030300 | 2025 | 4     | GP    | erie          | precipitation | 149.63345633705637 | 1481.0363897867212 |
# | 2025030300 | 2025 | 4     | GP    | michigan-huron| precipitation | 111.958298685779   | 5067.19286023333 |
# 

# %%
df_long = df_pivoted.melt(
    id_vars=["cfs_run", "forecast_month", "model", "lake"],
    value_vars=["precipitation", "evaporation", "runoff", "nbs"],
    var_name="component",
    value_name="value [mm]"
)

df_long["forecast_month"] = pd.to_datetime(
    df_long["forecast_month"],
    format="%Y-%m"
)

# Determine the first forecast month and filter the forecast data because we don't need
# to include older forecasts that are in the CFS database.
first_forecast_month = get_first_forecast_month()

# Use .copy() after filtering
df_out = df_long.loc[df_long["forecast_month"] >= first_forecast_month].copy()

df_out["month"] = df_out["forecast_month"].dt.month
df_out["year"] = df_out["forecast_month"].dt.year

df_out = df_out[
    [
        "cfs_run",
        "month",
        "year",
        "model",
        "lake",
        "component",
        "value [mm]",
    ]
]

df_units = convert_mm_to_cms(df_out)

# Save the final output to a CSV file
df_units.to_csv(input_dir + "forecast/CNBS_forecast.csv", sep="\t", index=False)

# %% [markdown]
# This cell saves the pivoted forecast DataFrame (`df_pivoted`) to a SQLite database for structured storage and easy retrieval.  
# 
# - Using a database allows efficient queries, aggregation, and integration with other datasets, enabling downstream analysis and plotting without repeatedly reprocessing the raw forecast data.

# %%
CFSDatabase(cnbs_database, cnbs_table).add_df(df_pivoted)

# %% [markdown]
# ### Below are other Optional Formats to save Data ###

# %% [markdown]
# This cell filters the pivoted forecast DataFrame to only include forecasts for the **first valid forecast month** and after, as determined by the `get_first_forecast_month()` function.
# 
# **Logic for determining the first forecast month:**
# - If today’s date is **before the 26th** → use the **current month**.
# - If today’s date is **on or after the 26th** → use the **next month**.
# 
# The `filter` function applies this filter to `df_pivoted`, returning only rows where the `forecast_month` matches the calculated first month and after.
# 
# For verification purposes, the function also prints the first forecast month in the format: YYYY-MM.

# %% [markdown]
# **Optional Step — Calculate Mean by Model and Forecast Month**
# 
# This step computes the **mean value of each forecast component** (e.g., precipitation, evaporation, runoff, NBS) for each combination of `model` and `forecast_month`.
# 
# **Process:**
# - Group the filtered forecast DataFrame (`df_filtered`) by `model` and `forecast_month`.
# - Calculate the mean of all numeric columns within each group (`numeric_only=True` ensures non-numeric columns are ignored).
# - Return a summarized DataFrame (`df_mean`) where each row represents the average forecast values for a given model and forecast month.
# 
# This aggregated dataset is useful for:
# - Comparing models at the monthly scale.
# - Generating cleaner plots by removing run-to-run noise.
# - Preparing data for ensemble analysis.

# %%
df_filtered = ForecastTransformer(df_pivoted).filter(first_forecast_month)
#df_mean = (
#    df_filtered
#      .groupby(["model", "forecast_month", "lake"], as_index=False)
#      .mean(numeric_only=True)
#)

# Save to a CSV [mm]
#df_mean.to_csv(input_dir + 'forecast/CNBS_model_mean_forecast.csv', sep='\t', index=False)

# %% [markdown]
# **Optional Step — Calculate Overall Mean Across All Model Ensembles**
# 
# This step calculates the **overall mean forecast** for each `forecast_month` by combining results from **all model ensembles**.
# 
# **Process:**
# - Group the filtered forecast DataFrame (`df_filtered`) by `forecast_month` only, ignoring model differences.
# - Compute the mean of all numeric columns (`numeric_only=True`), rounding values to three decimal places for cleaner output.
# - Save the resulting dataset (`df_all_mean`) as a tab-delimited CSV file (`CNBS_forecast.csv`) for record-keeping or sharing.
# 
# **Purpose:**
# - Provides a **single ensemble mean forecast** for each month.
# - Useful for quick summaries, reports, or when model-level detail is not required.

# %%
#df_all_mean = (
#    df_filtered
#      .groupby(["forecast_month"], as_index=False)
#      .mean(numeric_only=True).round(3)
#)

# Save to a CSV [mm]
#df_all_mean.to_csv(input_dir + 'forecast/CNBS_overall_mean_forecast.csv', sep='\t', index=False)

# %% [markdown]
# ## Climatology Exceedance Probability
# 
# **Climatology Exceedance Probability (CEP)**, also referred to as **Probability of Exceedance (POE)** by USACE, is the fraction of historical climatological values that exceed a given forecast value for a specific variable, location, and time period. It quantifies where the forecast falls within the climatological distribution and is calculated as the probability that a climatological value is greater than the forecast value.
# 
# A higher DEI indicates a drier-than-normal forecast (lower forecast values relative to climatology), while a lower DEI indicates a wetter-than-normal forecast (higher forecast values relative to climatology).
# 
# Interpretation:
# 
# - **CEP > 0.5** → Forecast indicates drier-than-normal conditions
# - **CEP = 0.5** → Forecast is near climatology (normal conditions)
# - **CEP < 0.5** → Forecast indicates wetter-than-normal conditions

# %% [markdown]
# ### Calculate Climatology Exceedance Probabilities (CEP)
# 
# This cell computes the **Climatology Exceedance Probability (CEP)**, aka **POE** for each forecasted Net Basin Supply (NBS) value.
# 
# First, the monthly climatology probability lookup tables for each lake are loaded in millimeters. These lookup tables contain climatological NBS values associated with exceedance probabilities for each calendar month.
# 
# The `CEPCalculator` class is then initialized using the probability dataset and aligned to the **first forecast month**, ensuring each climatology month corresponds to the correct forecast period.
# 
# Finally, the CEP is calculated for each forecast in `df_filtered` by comparing the forecasted NBS value to the climatological distribution and finding the nearest climatological value. The associated exceedance probability is returned as the CEP.
# 
# The resulting dataframe is saved to:
# `forecast/climatology_exceedance_probabilities.csv`

# %%
prob = DataLoader().lake_probabilities(probability_dir, units="mm")

cep_calc = CEPCalculator(prob)

cep_calc.align_prob_with_start_date(first_forecast_month)

df_cep = cep_calc.calculate_cep(
    df=df_filtered,
    output_file=input_dir + 'forecast/climatology_exceedance_probabilities.csv'
)


