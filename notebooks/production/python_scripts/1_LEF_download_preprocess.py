# %% [markdown]
# # Step 1: Download/Pre-process CFS forecast data
# Lindsay Fitzpatrick
# ljob@umich.edu
# 
# **Initial Version:** 08/28/2024  
# **Last Updated:** 09/16/2025 
# 
# This script automates the download and preprocessing of **Climate Forecast System (CFS) version 2** forecast data for the Great Lakes region. The data is sourced in GRIB2 format from either **AWS** or **NCEI** repositories.
# 
# ### Key Functionality
# - Downloads raw CFS forecast files for specified dates and forecast hours.
# - Processes the GRIB2 files to calculate key atmospheric metrics:
#   - Total precipitation
#   - Evaporation
#   - 2-meter air temperature (averaged separately over lake and land surfaces)
# - Applies a geographic mask to isolate values over lake and land areas using `GL_mask.nc`.
# - Saves the processed results into a local **SQLite database** (`cfs_forecast_data.db`), either by appending to an existing file or creating a new one.
# 
# ### Required Input Files
# - **`GL_mask.nc`**  
#   A NetCDF file containing geographic masks to distinguish lake and land grid cells in the Great Lakes region.
# 
# - **`cfs_forecast_data.db`** *(optional)*  
#   If provided, the script will append new forecast data to this database; otherwise, it will create a new database with the appropriate schema.
# 
# This script forms the first step in the CNBS forecasting pipeline, ensuring that clean, lake-specific forecast variables are prepared for use in downstream modeling and prediction.

# %%
import os
import sys
# Add the path to the src directory (two levels up)
sys.path.append(os.path.abspath('../../'))
from src.data_downloader import CFSDownloader
from src.data_processor import CFSProcessor
from src.database_utils import CFSDatabase

# %% [markdown]
# ## User Inputs
# ### Configuration: File Paths and Processing Options
# 
# This section defines key file paths and user-configurable options for downloading and processing CFS forecast data:
# 
# - **`local_path`**: Base directory where the repository is cloned and data folders are located.  
# - **`download_dir`**: Directory where raw CFS GRIB2 files will be downloaded and temporarily stored.  
# - **`input_dir`**: Directory containing required input files such as masks and scalers.  
# - **`mask_file`**: Path to the Great Lakes mask file (`GL_mask.nc`) used for separating lake and land data.  
# - **`database`**: Path to the SQLite database where processed CFS forecast data is stored.  
# 
# #### Data Source and Processing Control  
# - **`source`**: Specifies the data source, either `'aws'` or `'ncei'`.  
# - **`download_cfs`**: Toggle to enable or disable downloading new CFS data (`'yes'` or `'no'`).  
# - **`process_cfs`**: Toggle to enable or disable processing of downloaded CFS data (`'yes'` or `'no'`).  
# - **`delete_files`**: Option to delete raw GRIB2 files after processing to save storage (`'yes'` or `'no'`).  
# 
# #### Date Range Configuration  
# - **`auto`**: When set to `'yes'`, automatically updates the database by detecting the last processed date and downloading data up to yesterday’s date.  
# - If **`auto`** is `'no'`, specify manual start and end dates for data download and processing:  
#   - **`start_date`**: Starting date for data retrieval (format: MM-DD-YYYY).  
#   - **`end_date`**: Ending date for data retrieval (format: MM-DD-YYYY).  
# 
# These settings allow flexible control over data acquisition and processing, supporting both automated updates and manual testing or reprocessing.

# %%
# Directory where the repository is cloned
local_path = '/Users/ljob/Desktop/'

# Path to data directory
input_dir = local_path + 'cnbs-predictor/data/'

# Path the GL mask file
mask_file = input_dir + 'input/GL_mask.nc'

# Path to save downloaded data
download_dir = input_dir + 'cfs/'

# Path to the CFS forecast database
database = local_path + 'cfs_forecast_data.db'
table = 'cfs_forecast_data'

# Data source: specify either 'aws' or 'ncei'
source = 'aws'

# Do you need to download CFS data? ('yes' or 'no')
download_cfs = 'yes'

# Do you want to process the CFS data? ('yes' or 'no')
process_cfs = 'yes'

# Should grib files be deleted after processing? ('yes' or 'no')
delete_files = 'no'

# Auto mode will automatically open the existing database, pull the last entered date to determine the start date, 
# and set the end date to yesterday, making the database 'up-to-date'. If 'no', you can manually enter a start and 
# end date (ideal for testing or if you need to redownload/reprocess specific time frames).
auto = 'yes'

# Specify the start and end dates if auto mode above is set to 'no' (Format: MM-DD-YYYY)
start_date = '05-22-2026'
end_date = '05-23-2026'

# %% [markdown]
# ### Preset Variables
# 
# This section defines key preset variables used throughout the CFS data download and processing workflow:
# 
# - **`products`**: List of CFS forecast product types to retrieve.  
#   - `'pgb'`: Pressure-level forecast data  
#   - `'flx'`: Surface flux data
# 
# - **`utc`**: List of UTC initialization hours for which forecasts will be downloaded (`00`, `06`, `12`, `18`).
# 
# - **`mask_variables`**: Names of the lake and land regions used for applying the spatial mask.  
#   Each entry corresponds to a specific Great Lake (e.g., Erie, Ontario, Michigan-Huron, Superior), with distinctions for lake surface and surrounding land areas.
# 
# - **`bucket_name`**: AWS S3 bucket name (`noaa-cfs-pds`) used to access the public NOAA CFS forecast data.
# 
# These presets standardize the data sources, time intervals, and spatial definitions used in the forecast extraction and preprocessing steps.

# %%
## Presets ##
products = ['pgb','flx']
utc = ['00','06','12','18']

# Define mask variables
mask_variables = ['eri_lake','eri_land',
                  'ont_lake','ont_land',
                  'mih_lake','mih_land',
                  'sup_lake','sup_land']

#AWS bucket name to locate the CFS forecast
bucket_name = 'noaa-cfs-pds'

# %% [markdown]
# ## Begin Script

# %% [markdown]
# Create a database handler.

# %%
db = CFSDatabase(database, table)

# %% [markdown]
# ### Determine Date Range for Forecast Download and Processing
# 
# This section defines the date range over which the script will operate, based on the user's selected mode:
# 
# - When **`auto = 'yes'`**:  
#   The script automatically checks the `cfs_forecast_data.db` database for the most recent recorded CFS run.  
#   - The **start date** is set to the day after the last recorded run.  
#   - The **end date** is set to **yesterday**, ensuring the database stays up to date with the latest available forecasts.
# 
# - When **`auto = 'no'`**:  
#   The user manually specifies both the start and end dates.  
#   This is especially useful for testing or initializing a new database with historical forecasts.
# 
# Before proceeding, the script prints a message confirming the selected date range, for example:  
# `Starting from: 05-22-2025 00Z and continuing through: 06-10-2025 18Z`
# 
# This ensures transparency and gives the user a final opportunity to confirm the configured date range before data download and processing begins.

# %%
start_date, end_date, date_array = db.get_date_range(auto, start_date, end_date)

# %% [markdown]
# ### Loop Through Forecast Dates: Download, Process, and Store Data
# 
# This section iterates over each date in the user-defined `date_array` to handle CFS forecast data for the Great Lakes using dedicated modules and functions.
# 
# For each date:
# 
# 1. **Prints progress** to the console (e.g., "Beginning Files for 2025-06-10").
# 
# 2. **Download CFS forecast files** (`download_cfs = 'yes'`):  
#    - Uses the `CFSDownloader` module to fetch forecast data.  
#    - The `download()` function automatically constructs the correct file paths based on the chosen source (`aws` or `ncei`) and downloads all relevant GRIB2 files for that date.
# 
# 3. **Process downloaded GRIB2 files** (`process_cfs = 'yes'`):  
#    - The `CFSProcessor` module reads GRIB2 files from the daily subdirectory.  
#    - `process_files()` calculates atmospheric metrics (e.g., precipitation, evaporation, temperature) and applies spatial masks.  
#    - Results are saved to the configured SQLite database (`cfs_forecast_data.db`) under the `cfs_forecast_data` table.
# 
# 4. **Optional cleanup** (`delete_files = 'yes'`):  
#    - Deletes the local GRIB2 directory for that date to conserve storage space.  
#    - If the directory cannot be deleted, a warning is printed and processing continues.
# 
# After all dates are processed, the script prints:  
# **"Process Complete"**
# 
# This modular loop ensures a clear, maintainable pipeline for downloading, processing, and storing CFS forecast data efficiently using the new `CFSDownloader` and `CFSProcessor` classes.

# %%
for date in date_array[:-1]:
    print(f"Beginning Files for {date}.")

    date_str = date.strftime('%Y%m%d%H')
    YYYY, MM, DD, HH = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'), date.strftime('%H')

    # ===== Download CFS Forecast Files =====
    if download_cfs.lower() == 'yes':
        start_date = date.strftime('%m-%d-%Y')

        cfs_downloader = CFSDownloader()

        cfs_downloader.download(
            download_dir, 
            start_date,
            end_date=None,
            products=products,
            source=source,
            )

    # ===== Process GRIB2 Files =====
    
    if process_cfs.lower() == 'yes':
        date_str = date.strftime('%Y%m%d')
        download_path = f'{download_dir}{date_str}/'
        
        processor = CFSProcessor(database=database, table=table)

        processor.process_files(
            download_path, 
            mask_file, 
            mask_variables
            )

        if delete_files.lower() == 'yes':
            try:
                os.rmdir(download_path)
            except OSError:
                print(f"Cannot delete directory {download_path}. Skipping.")
    
    print(f'Done with {date}.')
print("***** Process Complete *****")


