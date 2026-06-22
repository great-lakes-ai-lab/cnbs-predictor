# %% [markdown]
# # Step 3: Forecast Visualization Outputs
# Lindsay Fitzpatrick
# ljob@umich.edu
# 
# **Initial Version:** 06/16/2026  
# 
# This script reads in forecast data generated from **2_LEF_forecast_model** and saved in the **cfs_forecast_data.db** database. It creates several figures to make the 12-month forecast easier and faster interpretation.
# 
# ### Plot 1: Static 12-Month Forecast Time Series
# 
# Creates a static 4 × 4 time series figure showing the 12-month forecast for each lake and forecast component.
# 
# - **Columns:** precipitation, evaporation, runoff, and total NBS
# - **Rows:** Superior, Michigan-Huron, Erie, and Ontario
# - **Units:** millimeters `[mm]`
# - **Colored lines:** mean forecast value from each model
# - **Shaded region:** 95% confidence interval calculated across all models
# 
# ### Plot 2: Interactive 12-Month Forecast Time Series
# 
# Creates the same 4 × 4 forecast figure as **Plot 1**, but in an interactive HTML format. This version allows users to hover over each time step to view the forecast month, model name, and forecast value.
# 
# ### Plot 3: Probability of Exceedance Time Series
# 
# Calculates and plots the probability of exceedance using climatology as the reference condition.
# 
# - Each line represents one model
# - A probability of **0.5** represents climatology
# - Values above 0.5 indicate wetter-than-normal conditions
# - Values below 0.5 indicate drier-than-normal conditions
# 
# ### Plot 4: Spatial Probability of Exceedance Maps
# 
# Creates a 3 × 4 spatial figure showing probability of exceedance across the Great Lakes for each forecast month.
# 
# - Each subplot represents one forecast month
# - Lakes are shaded based on their probability of exceedance value
# - A diverging color scale is used:
#   - **Brown:** drier conditions
#   - **Blue:** wetter conditions
#   - **Near neutral:** close to climatology
# 
# ### Required Input Files
# - **`cnbs_forecast_data.db`**
#   This file holds the forecast for **precipitation**, **evaporation**, **runoff**, and **net basin supply** created from **`Script 2`**.
# 
# - **`climatology_exceedance_probability.csv`**
# 
#   This file contains **net basin supply** and the cooresponding **climatology exceedance probability** calculated in **`Script 2`** using the **probability of exceedance** files from USACE.

# %%
import pandas as pd
import os
import sys

# Add the path to the src directory (two levels up)
sys.path.append(os.path.abspath('../../'))
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

# Path to file with CED (or POE) values
probability_dir = input_dir + 'probabilities/'
prob_file = input_dir + 'forecast/climatology_exceedance_probabilities.csv'

# %% [markdown]
# ### Define Forecast Start Date
# 
# This step determines the **first valid forecast month** using the `get_first_forecast_month()` function.
# 
# The returned `start_date` is used to filter out any forecast months that have already passed relative to the current date. Removing outdated forecast periods improves script efficiency by reducing unnecessary processing and ensures that only future-relevant forecast data is retained for analysis and visualization.
# 
# The script will also print:
# 
# `First forecast month: YY-MMM`
# 
# This serves as a quick verification that the forecast start date is correct and corresponds to **forecast month 0 (mo0)** in the model output.

# %%
start_date = get_first_forecast_month()

# %% [markdown]
# ### Load and Filter Forecast Data
# 
# This step loads the CNBS forecast output from the forecast database and filters the data to include only valid forecast months.
# 
# - **Load forecast data**  
#   The `CFSDatabase` class reads the CNBS forecast results from the SQLite database (`cnbs_forecast.db`) and loads them into a dataframe.
# 
# - **Filter forecast months**  
#   The `ForecastTransformer.filter()` function removes any forecast records that occur before the defined `start_date` (forecast month 0). This ensures that only current and future forecast periods are retained.
# 
# The resulting dataframe, `df_filtered`, contains the cleaned forecast data used for subsequent analysis, visualization, and Probability of Exceedance calculations.

# %%
# Load CNBS predictions
data = CFSDatabase(cnbs_database, cnbs_table).load()
df_filtered = ForecastTransformer(data).filter(start_date)

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
fig1 = plot_cnbs_forecast(df_filtered, input_dir + 'forecast/figures/CNBS_forecast.png')

# %% [markdown]
# ## Plot 2
# ### **Interactive 4×4 Grid of Great Lakes 12-Month NBS Forecasts**
# 
# This section uses **Plotly** to create a fully interactive **4×4 subplot grid** that visualizes the mean forecasts and uncertainty ranges for precipitation, evaporation, runoff, and total CNBS across the four Great Lakes basins.

# %%
fig2 = plot_cnbs_forecast_interactive(df_filtered, input_dir + 'forecast/figures/CNBS_forecast.html')

# %% [markdown]
# ## Plot 3
# 
# ### **NBS forecast with Climatology**
# 
# This section focuses on the NBS forecast, same as above, but adds the climatology for an easy visual of whether the forecast is expected to be above or below climatology.

# %%
prob = DataLoader().lake_probabilities(probability_dir, units="mm")
prob_aligned = CEPCalculator(prob).align_prob_with_start_date(start_date)

# %%
fig3 = plot_nbs_forecast(df_filtered, prob_aligned, input_dir + 'forecast/figures/NBS_forecast.png')

# %% [markdown]
# ## Plot 4
# ### Plot Climatology Exceedance Probability Time Series
# 
# This step generates the **Climatology Exceedance Probability (CEP) time series plot** and saves it as a PNG file.
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

# %%
fig4 = plot_cep_timeseries(df_cep, input_dir + 'forecast/figures/cep_timeseries.png')

# %% [markdown]
# ## Plot 5
# ### Plot **Climatology Exceedance Probability (CEP)** Spatial Maps
# 
# **Climatology Exceedance Probability (CEP)** is the fraction of historical climatological values that exceed a given forecast value for a specific variable, location, and time period. It quantifies where the forecast falls within the climatological distribution and is calculated as the probability that a climatological value is greater than the forecast value.
# 
# A higher DEI indicates a drier-than-normal forecast (lower forecast values relative to climatology), while a lower DEI indicates a wetter-than-normal forecast (higher forecast values relative to climatology).
# 
# CEP = 0.95 (95%) → 95% of climatology is greater than the forecast → very dry
# CEP = 0.50 (50%) → Forecast is near climatological median → near normal
# CEP = 0.05 (5%) → Only 5% of climatology is greater than the forecast → very wet
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
fig5 = plot_cep_spatial(
    df_cep=df_cep,
    model="RF",
    value_col="cep",
    filename=input_dir + 'forecast/figures/climatology_exceedance_spatial_RF.png'
)


