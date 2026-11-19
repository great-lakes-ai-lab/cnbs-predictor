# %% [markdown]
# # Step 3: Forecast Visualization Outputs
# Lindsay Fitzpatrick
# ljob@umich.edu
# 
# **Initial Version:** 06/16/2026
# **Updated:** 07/27/2026 
# 
# This script reads in forecast data generated from **2_LEF_forecast_model** and saved in the **cfs_forecast.db** database. It creates several figures to make the 12-month forecast easier and faster interpretation.
# 
# ### Required Files
# 
# - **`cnbs_forecast.db`**
#   This file holds the forecast for **precipitation**, **evaporation**, **runoff**, and **net basin supply** created from **`Script 2`**.
# 
# - **`climatology_exceedance_probability.csv`**
# 
#   This file contains **net basin supply** and the cooresponding **climatology exceedance probability** calculated in **`Script 2`** using the **probability of exceedance** files from USACE.
#   
# ### Required Forecast DataFrame Format
# 
# The plotting functions require the **`cnbs_forecast.db`** to be in **long format**. Each row should represent a single forecast value for a given:
# 
# - CFS forecast initialization time (`cfs_run`)
# - forecast month (`forecast_month`)
# - model
# - lake
# - forecast component
# 
# Example dataframe:
# 
# | cfs_run | forecast_month | model | lake | component | value [mm] | value [cms] |
# |---------|----------------|-------|------|-----------|------------|-------------|
# | 2025101700 | 2026-07 | GP | erie | precipitation | 50.89 | 487.45 |
# | 2025101700 | 2026-07 | GP | erie | evaporation | -42.13 | -404.12 |
# | 2025101700 | 2026-07 | GP | erie | runoff | 20.15 | 193.10 |
# | 2025101700 | 2026-07 | GP | erie | nbs | 28.91 | 276.43 |
# | 2025101700 | 2026-08 | GP | erie | precipitation | 98.24 | 940.96 |
# 
# ### Notes
# 
# - `forecast_month` should be stored as a string in `YYYY-MM` format (e.g., `2026-07`).
# - The dataframe should **not** be pivoted. Components should remain in the `component` column rather than being separate columns.
# - The plotting functions automatically select either:
#   - `value [mm]` when `units="mm"` (default)
#   - `value [cms]` when `units="cms"`
# - Multiple models and ensemble members are supported. The functions calculate the ensemble mean and 95% range from all available model forecasts.
# 
# ### Plot 1: 12-Month Forecast Time Series
# 
# Creates a static 4 × 4 time series figure showing the 12-month forecast for each lake and forecast component.
# 
# - **Columns:** precipitation, evaporation, runoff, and total NBS
# - **Rows:** Superior, Michigan-Huron, Erie, and Ontario
# - **Units:** millimeters `[mm]`
# - **Colored lines:** mean forecast value from each model
# - **Dashed black line:** climatology calculated from L2SWBM
# - **Shaded region:** 95% confidence interval calculated across all ensemble members and models
# 
# ### Plot 2: Interactive 12-Month Forecast Time Series
# 
# Creates the same 4 × 4 forecast figure as **Plot 1**, but in an interactive HTML format. This version allows users to hover over each time step to view the forecast month, model name, and forecast value.
# 
# ### Plot 3: 12-Month Forecast Time Series for NBS
# 
# Creates a static 1 × 4 time series figure focusing on the 12-month NBS forecast for each lake.
# 
# - **Rows:** Superior, Michigan-Huron, Erie, and Ontario
# - **Units:** millimeters `[mm]`
# - **Colored lines:** mean forecast value from each model
# - **Dashed black line:** climatology calculated from L2SWBM
# - **Shaded region:** 95% confidence interval calculated across all ensemble members and models
# 
# ### Plot 4: 12-Month Cumulative Forecast Time Series
# 
# Creates a 4 × 4 time series figure showing the cumulative 12-month forecast for each lake and forecast component. Similar to **Plot 1**, this figure cumulatively sums the monthly values to illustrate how deviations from climatology accumulate throughout the forecast year and influence each lake’s overall NBS.
# 
# - **Columns:** precipitation, evaporation, runoff, and total NBS
# - **Rows:** Superior, Michigan-Huron, Erie, and Ontario
# - **Units:** millimeters `[mm]`
# - **Colored lines:** mean forecast value from each model
# - **Dashed black line:** climatology calculated from L2SWBM
# - **Shaded region:** 95% confidence interval calculated across all ensemble members and models
# 
# ### Plot 5: Probability of Exceedance Time Series
# 
# Calculates and plots the probability of exceedance using climatology as the reference condition.
# 
# - Each line represents one model
# - A probability of **0.5** represents climatology
# - Values **above 0.5** indicate wetter-than-normal conditions
# - Values **below 0.5** indicate drier-than-normal conditions
# 
# ### Plot 6: Spatial Probability of Exceedance Maps
# 
# Creates a 3 × 4 spatial figure showing probability of exceedance across the Great Lakes for each forecast month.
# 
# - Each subplot represents one forecast month
# - Lakes are shaded based on their probability of exceedance value
# - A diverging color scale is used:
#   - **Brown:** drier conditions
#   - **Blue:** wetter conditions
#   - **Near neutral:** close to climatology

# %%
import pandas as pd
import os
import sys

# Add the path to the src directory (two levels up)
sys.path.append(os.path.abspath('../../../'))
from src.data_processor import *
from src.database_utils import CFSDatabase
from src.data_loader import DataLoader
from src.plotting import *
from src.utilities import get_first_forecast_month

# %% [markdown]
# ### Define File Paths
# 
# This section defines the local directory paths used throughout the forecasting and visualization workflow.
# 
# - **`local_path`**: Root directory where the CNBS-Predictor repository is cloned.
# - **`input_dir`**: Path to the main data directory containing forecast inputs, output files, and supporting datasets.
# - **`cnbs_database`**: Location of the SQLite database used to store forecast results.
# - **`cnbs_table`**: Name of the database table containing CNBS forecast output.
# - **`probability_dir`**: Directory containing climatological probability files used for Probability of Exceedance (POE) calculations.
# 
# Updating these paths allows the script to locate input data and save forecast outputs correctly based on the user's local environment.

# %%
# Directory where the repository is cloned
local_path = '/Users/ljob/Desktop/'

# Path to data directory
input_dir = local_path + 'cnbs-predictor/data/'

# Path to save CNBS forecast output to a database
cnbs_database = input_dir + 'forecast/cnbs_forecast.db'
cnbs_table = 'cnbs_forecast'

# Path to Climatology directory
clim_dir = input_dir + "input/climatology/"

# Path to file with CED (or POE) values
probability_dir = input_dir + 'probabilities/'
prob_file = input_dir + 'forecast/climatology_exceedance_probabilities.csv'

# %% [markdown]
# ### Load Forecast Data
# 
# This step loads the CNBS forecast output from the forecast database.
# 
# - **Load forecast data**  
#   The `CFSDatabase` class reads the CNBS forecast results from the SQLite database (`cnbs_forecast.db`) and loads them into a dataframe. Passing `start_date` filters the records in SQL, so only the forecast months actually needed are read from the database instead of the whole table. This matters most when the database is on a shared network drive, where transferring the full table dominates the runtime.

# %%
# Load CNBS predictions
df = CFSDatabase(cnbs_database, cnbs_table).load()

# %% [markdown]
# ### Load Climatology Data
# 
# This step loads the precipitation, evaporation, runoff, and NBS climatology data calculated from the Large Lake Statistical Water Balance Model (L2SWBM) data.

# %%
climatology = SeasonalCycleProcessor()
scp_X, scp_y = climatology.load_clim(clim_dir)
clim_df = scp_y.climatology

# %% [markdown]
# ## Plot 1
# ### **Plotting the 12-Month Great Lakes Net Basin Supply (NBS) Forecasts**
# 
# This cell generates a **4×4 grid of subplots** visualizing precipitation, evaporation, runoff, and total NBS forecasts for each of the four Great Lakes basins (Superior, Michigan-Huron, Erie, Ontario).  
# 
# #### Key Steps:
# 
# 1. **Color Assignment**  
#    - Assigns a distinct color to each model using a predefined color list (`custom_colors`).  
#    - Raises an error if the number of models exceeds available colors.
# 
# 2. **Data Setup**  
#    - Defines the plotting order of 16 forecast variables (4 components × 4 lakes).  
#    - Creates subplot grid (`fig, axs`) for visual arrangement.  
#    - Tracks separate y-axis ranges for precipitation, evaporation, and runoff (columns 0–2) versus total NBS (column 3).
# 
# 3. **Plotting for Each Variable**  
#    - For each component/lake combination:
#      - Computes **95% confidence intervals** using the 2.5th and 97.5th percentiles.
#      - Plots the mean forecast for each model with unique colors and markers.
#      - Shades the 95% confidence interval region in gray (`fill_between`).
#      - Adds horizontal zero reference lines, gridlines, and shared y-ticks.
#      - Dynamically adjusts y-axis limits so that subplots share consistent scaling.
# 
# 4. **Formatting**  
#    - Adds component titles across the top row (Precipitation, Evaporation, Runoff, Total NBS).
#    - Labels y-axes by lake name for the first column.
#    - Formats x-axis dates (`%m-%Y`) for the bottom row, rotating labels for readability.
# 
# 5. **Final Layout & Output**  
#    - Synchronizes y-axis limits across similar variable types.
#    - Sets the plot’s main title: *"Great Lakes 12-Month CNBS [mm] Forecast"*.
#    - Adds a legend to the top-right subplot.
#    - Saves the figure as `CNBS_forecast.png` in the forecast output directory.
#    - Displays the final figure.
# 
# **Purpose:**  
# This visualization provides a **multi-model ensemble view** of the Great Lakes’ hydrological forecast components over the next 12 months, allowing easy comparison of models, seasonal cycles, and uncertainty ranges.

# %%
fig1 = plot_cnbs_forecast(
    df, 
    date_col=["year","month"], 
    units="mm", 
    filename=input_dir + "forecast/figures/cnbs_forecast.png",
    climatology=clim_df)

# %% [markdown]
# ## Plot 2
# ### **Interactive 4×4 Grid of Great Lakes 12-Month NBS Forecasts**
# 
# This section uses **Plotly** to create a fully interactive **4×4 subplot grid** that visualizes the mean forecasts and uncertainty ranges for precipitation, evaporation, runoff, and total CNBS across the four Great Lakes basins.

# %%
fig2 = plot_cnbs_forecast_interactive(
    df, 
    date_col=["year","month"],
    units="mm",
    filename=input_dir + 'forecast/figures/cnbs_forecast.html',
    climatology=clim_df
    )

# %% [markdown]
# ## Plot 3
# 
# ### **NBS forecast with Climatology**
# 
# This section focuses on the NBS forecast, same as above, but adds the climatology for an easy visual of whether the forecast is expected to be above or below climatology.

# %%
start_date = get_first_forecast_month()
prob = DataLoader().lake_probabilities(probability_dir, units="mm")
prob_aligned = CEPCalculator(prob).align_prob_with_start_date(start_date)

# %%
fig3 = plot_nbs_forecast(
    df, 
    prob_aligned,
    date_col=["year","month"],
    units="mm", 
    filename=input_dir + 'forecast/figures/nbs_forecast.png'
    )

# %% [markdown]
# ## Plot 4
# 
# ### **Cumulative CNBS forecast with Climatology**
# 
# This figure plots the NBS components and NBS as cumulative values and adds the climatology for an easy visual of whether the forecast is expected to be above or below climatology.

# %%
fig4 = plot_cumulative_cnbs_forecast(
    df,
    units="mm",
    date_col=["year","month"],
    filename=input_dir + 'forecast/figures/cumulative_forecast.png',
    climatology=clim_df
)

# %% [markdown]
# ## Plot 5
# ### Plot **Climatology Exceedance Probability (CEP)** Time Series
# 
# **Climatology Exceedance Probability (CEP)**, or Probability of Exceedance (POE) as it it known by the USACE, is the fraction of historical climatological values that exceed a given forecast value for a specific variable, location, and time period. It quantifies where the forecast falls within the climatological distribution and is calculated as the probability that a climatological value is greater than the forecast value.
# 
# A higher CEP indicates a drier-than-normal forecast (lower forecast values relative to climatology), while a lower DEI indicates a wetter-than-normal forecast (higher forecast values relative to climatology).
# 
# The `plot_cep_timeseries()` function takes the calculated CEP dataframe (`ced_df`) and creates a visualization showing how each model compares to climatological conditions over the forecast period.
# 
# Plot characteristics:
# 
# - **X-axis:** Forecast month (12-month outlook)
# - **Y-axis:** Climatology Exceedance Probability (0–1)
# - **Lines:** Individual model forecasts
# - **Reference line at 0.5:** Climatological normal
# 
# Interpretation:
# 
# - **CEP = 0.5** → Forecast is near climatology (normal conditions)
# - **CEP > 0.5** → Forecast indicates drier-than-normal conditions
# - **CEP < 0.5** → Forecast indicates wetter-than-normal conditions
# 
# This figure provides a quick way to compare model agreement and assess whether the forecast is trending toward unusually wet or dry conditions. Large separation between model lines indicates higher forecast uncertainty, while close agreement suggests greater confidence.
# 
# The figure is saved to:
# 
# `forecast/figures/ced_timeseries.png`

# %%
df_cep = pd.read_csv(prob_file, sep=',')
fig5 = plot_cep_timeseries(df_cep, input_dir + 'forecast/figures/cep_timeseries.png')

# %% [markdown]
# ## Plot 6
# ### Plot **Climatology Exceedance Probability (CEP)** Spatial Maps
# 
# **Climatology Exceedance Probability (CEP)** is the fraction of historical climatological values that exceed a given forecast value for a specific variable, location, and time period. It quantifies where the forecast falls within the climatological distribution and is calculated as the probability that a climatological value is greater than the forecast value.
# 
# The `plot_cep_spatial_12month()` function creates a **3 × 4 panel figure**, where each subplot represents one forecast month in the 12-month outlook. In this example, the plot is generated using the **Random Forest (RF)** model but can be set based on the user.
# 
# Function inputs:
# 
# - **`cep_df`**: Dataframe containing calculated CEP values for each lake, forecast month, and model
# - **`model="RF"`**: Selects the Random Forest model for visualization
# - **`filename`**: Output file path for saving the figure
# 
# Plot characteristics:
# 
# - **Layout:** 3 rows × 4 columns (12 forecast months)
# - **Each subplot:** One forecast month
# - **Shaded regions:** Great Lakes colored by POE anomaly
# - **Color scale:** Diverging brown-to-blue colormap
# 
# Color interpretation:
# 
# - **Dark brown (high values)** → Drier-than-normal conditions  
# - **White / neutral** → Near climatological normal  
# - **Dark blue (low values)** → Wetter-than-normal conditions   
# 
# This spatial plot provides a quick visual summary of how wet or dry conditions evolve across the Great Lakes over the forecast horizon and helps identify lake-specific anomalies and regional patterns.
# 
# The figure is saved to:
# 
# `forecast/figures/climatology_exceedance_spatial_RF.png`

# %%
fig6 = plot_cep_spatial(
    df_cep=df_cep,
    model="RF",
    value_col="cep",
    filename=input_dir + 'forecast/figures/climatology_exceedance_spatial_RF.png'
)


