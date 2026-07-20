# # Step 1: Process CFS reforecast data
# Lindsay Fitzpatrick
# ljob@umich.edu
# **Last Updated:** 2026-07-16 

# This script was used on Umich's Great Lakes cluster to process the CFS reforecast data downloaded from NOAA's NCEI. 
# It reads the GRIB2 files, remaps the relevant fields (precipitation, 2-metre air temperature, evaporation from latent 
# heat flux) onto a basin mask grid, computes area-weighted lake/land averages, and writes the results to a SQLite database.
# CFS Reforecast data pulls Precipitation rate "prate" from the flux files because the pgb files have "tp" as 0 throughout.
# %%
import os
import sys
import netCDF4 as nc
import os
import pandas as pd
import cfgrib
import sqlite3
import numpy as np
import calendar
from datetime import datetime
import joblib
import json
import uuid
import re
from typing import Optional, Sequence, Dict
from src.data_downloader import CFSDownloader
from src.data_processor import CFSProcessor
from src.database_utils import CFSDatabase
from src.hydro_utils import calculate_evaporation_rate, calculate_grid_cell_areas

# Path the GL mask file
mask_file = '/home/ljob/bilsa/data/input/GL_mask.nc'

# Path to save downloaded data
download_dir = '/scratch/dannes_root/dannes0/ljob/cfs_reforecast/'

# Path to the CFS forecast database
database = '/scratch/dannes_root/dannes0/ljob/cfs_reforecast/cfs_reforecast_data_precip.db'
table = 'cfs_forecast_data'

# Data source: specify either 'aws' or 'ncei'
source = 'aws'

# Do you want to process the CFS data? ('yes' or 'no')
process_cfs = 'yes'

# Should grib files be deleted after processing? ('yes' or 'no')
delete_files = 'no'

# Auto mode will automatically open the existing database, pull the last entered date to determine the start date, 
# and set the end date to yesterday, making the database 'up-to-date'. If 'no', you can manually enter a start and 
# end date (ideal for testing or if you need to redownload/reprocess specific time frames).
auto = 'no'

# Specify the start and end dates if auto mode above is set to 'no' (Format: MM-DD-YYYY)
start_date = '12-12-1981'
end_date = '03-01-2011'

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

# %%
from pathlib import Path
from datetime import datetime

# Get start/end dates as before
start_date, end_date, _ = db.get_date_range(auto, start_date, end_date)

# Convert to datetime objects if they are strings
if isinstance(start_date, str):
    start_dt = datetime.strptime(start_date, "%m-%d-%Y")
else:
    start_dt = start_date

if isinstance(end_date, str):
    end_dt = datetime.strptime(end_date, "%m-%d-%Y")
else:
    end_dt = end_date

# Find all folders matching YYYYMMDD
date_array = []

for folder in Path(download_dir).iterdir():
    if folder.is_dir():
        try:
            folder_date = datetime.strptime(folder.name, "%Y%m%d")
            if start_dt <= folder_date <= end_dt:
                date_array.append(folder_date)
        except ValueError:
            # Skip folders that aren't YYYYMMDD
            continue

# Sort chronologically
date_array.sort()

# %%

class CFSProcessor:
    """Process downloaded CFS GRIB files into lake-averaged variables.

    Reads CFS forecast GRIB2 files for a run, remaps the relevant fields
    (precipitation, 2-metre air temperature, evaporation from latent heat
    flux) onto a basin mask grid, computes area-weighted lake/land averages,
    and writes the results to the forecast database via :class:`CFSDatabase`.

    Parameters
    ----------
    database : str
        Path to the SQLite database file used for output.
    table : str
        Name of the table to write processed values into.
    """

    def __init__(self, database, table):
        """
        Initialize the processor with database path and table name.
        """
        self.database = database
        self.table = table
        self.db = CFSDatabase(database, table)

    def process_files(self, download_dir, mask_file, mask_variables):
        """
        Process CFS GRIB files and insert lake-averaged variables into the database.

        This function loops through downloaded CFS GRIB files for a forecast run,
        extracts relevant variables, remaps them to the mask grid, calculates
        lake- or land-area weighted averages, and stores the results in the
        forecast database.

        Processed variables include:
            - precipitation from pgbf files
            - 2-meter air temperature from flxf files
            - evaporation from latent heat flux in flxf files

        Parameters
        ----------
        download_dir : str
            Directory containing downloaded CFS GRIB files.

        mask_file : str
            Path to the NetCDF mask file. The mask file must include:
                - latitude
                - longitude
                - lake/land mask variables listed in `mask_variables`

        mask_variables : list of str
            List of mask variable names to process.

            Expected format:
                "{lake_abbreviation}_{surface_type}"

            Examples:
                - "sup_lake"
                - "sup_land"
                - "mih_lake"
                - "eri_land"

            Valid lake abbreviations are:
                - "sup" -> "superior"
                - "mih" -> "michigan-huron"
                - "eri" -> "erie"
                - "ont" -> "ontario"

        Returns
        -------
        None
            Results are inserted directly into the database using `self.db.add()`.
        """

        # -------------------------
        # Validate inputs
        # -------------------------
        if not os.path.isdir(download_dir):
            raise ValueError("ERROR: The specified directory does not exist.")

        if not os.path.exists(mask_file):
            raise ValueError("ERROR: mask_file not found.")

        if not isinstance(mask_variables, list):
            raise ValueError("ERROR: mask_variables must be a list of strings.")

        # -------------------------
        # Load mask grid and areas
        # -------------------------
        # The mask grid defines the target latitude/longitude grid and lake/land masks.
        mask_ds = nc.Dataset(mask_file)

        mask_lat = mask_ds.variables["latitude"][:]
        mask_lon = mask_ds.variables["longitude"][:]

        # Calculate grid-cell area for area-weighted lake/land averages.
        area = calculate_grid_cell_areas(mask_lon, mask_lat)

        # Mapping from mask file lake abbreviations to full lake names.
        lake_lookup = {
            "eri": "erie",
            "ont": "ontario",
            "sup": "superior",
            "mih": "michigan-huron",
        }

        # -------------------------
        # Remove index files
        # -------------------------
        # cfgrib may create or use .idx files. Remove them so stale index files
        # do not interfere with reading newly downloaded GRIB files.
        for f in os.listdir(download_dir):
            if f.endswith(".idx"):
                os.remove(os.path.join(download_dir, f))

        # -------------------------
        # Process each GRIB file
        # -------------------------
        for filename in sorted(os.listdir(download_dir)):
            file = os.path.join(download_dir, filename)

            # File names are expected to contain the CFS run and forecast month.
            # Example structure depends on your downloaded CFS naming convention.
            parts = filename.split(".")

            cfs_run = str(parts[0][4:])
            print(cfs_run)

            forecast_year = int(parts[2][:4])
            forecast_month = int(parts[2][4:6])

            # Number of days in the forecast month, used to convert monthly totals.
            _, num_days = calendar.monthrange(forecast_year, forecast_month)

            # =====================================================
            # 2-meter air temperature and evaporation
            # =====================================================
            # flxf files contain temperature and latent heat flux fields.
            if filename.startswith("flxf") and filename.endswith(".grb2"):

                # -------------------------
                # 2-meter air temperature
                # -------------------------
                try:
                    # Open 2-meter height-above-ground fields.
                    flx_2mabove = cfgrib.open_dataset(
                        file,
                        engine="cfgrib",
                        filter_by_keys={
                            "typeOfLevel": "heightAboveGround",
                            "level": 2,
                        },
                        decode_timedelta=False,
                    )

                    # Variable name differs between CFS data versions.
                    try:
                        mean2t = flx_2mabove["avg_2t"]
                    except KeyError:
                        print("'avg_2t' not found in flux file, trying 'mean2t'.")
                        mean2t = flx_2mabove["mean2t"]

                    # Cut the temperature field to the mask extent.
                    mean2t_cut = mean2t.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max()),
                    )

                    # Remap temperature to the mask grid.
                    mean2t_remap = mean2t_cut.interp(
                        latitude=mask_lat,
                        longitude=mask_lon,
                        method="linear",
                    )

                    for mask_var in mask_variables:
                        # Create a mask where valid mask cells are retained.
                        mask = np.ma.masked_where(
                            np.isnan(mask_ds.variables[mask_var][:]),
                            np.ones_like(mask_ds.variables[mask_var][:]),
                        )

                        # Calculate mean 2-meter air temperature over the mask area.
                        tmp_avg = np.mean(mean2t_remap * mask)

                        lake_abv, surface_type = mask_var.split("_")
                        lake = lake_lookup.get(lake_abv)

                        if lake is None:
                            raise ValueError(
                                "ERROR: The mask variables need to begin with "
                                "'eri', 'ont', 'sup', or 'mih'. Check the mask file."
                            )

                        # Insert air temperature into the database.
                        self.db.add(
                            cfs_run,
                            forecast_year,
                            forecast_month,
                            lake,
                            surface_type,
                            "air_temperature",
                            tmp_avg.item(),
                        )

                except Exception as e:
                    print(f"ERROR processing temperature data: {e}. Skipping forecast.")
                    continue

                # -------------------------
                # Evaporation
                # -------------------------
                try:
                    # Open surface-level flux fields.
                    flx_surface = cfgrib.open_dataset(
                        file,
                        engine="cfgrib",
                        filter_by_keys={"typeOfLevel": "surface"},
                        decode_timedelta=False,
                    )

                    # Variable name differs between CFS data versions.
                    try:
                        mslhf = flx_surface["avg_slhtf"]
                    except KeyError:
                        print("'avg_slhtf' not found in flux file, trying 'mslhf'.")
                        mslhf = flx_surface["mslhf"]

                    # Cut latent heat flux to the mask extent.
                    mslhf_cut = mslhf.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max()),
                    )

                    # Remap latent heat flux to the mask grid.
                    mslhf_remap = mslhf_cut.interp(
                        latitude=mask_lat,
                        longitude=mask_lon,
                        method="linear",
                    )

                    # Convert latent heat flux to evaporation rate.
                    evap = calculate_evaporation_rate(mean2t_remap, mslhf_remap)

                    for mask_var in mask_variables:
                        mask = mask_ds.variables[mask_var][:]

                        # Calculate area-weighted monthly evaporation.
                        total_evap = np.sum(evap * area * mask) * num_days * 86400
                        evap_mm = total_evap / np.sum(mask * area)

                        lake_abv, surface_type = mask_var.split("_")
                        lake = lake_lookup.get(lake_abv)

                        if lake is None:
                            raise ValueError(
                                "ERROR: The mask variables need to begin with "
                                "'eri', 'ont', 'sup', or 'mih'. Check the mask file."
                            )

                        # Insert evaporation into the database.
                        self.db.add(
                            cfs_run,
                            forecast_year,
                            forecast_month,
                            lake,
                            surface_type,
                            "evaporation",
                            evap_mm.item(),
                        )

                    # -------------------------
                    # Precipitation
                    # -------------------------
                    pcp = flx_surface["prate"] #precipitation rate in kg/m^2/s

                    # Cut the precipitation field to the mask extent.
                    pcp_cut = pcp.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max()),
                    )

                    # Remap precipitation to the mask grid.
                    pcp_remap = pcp_cut.interp(
                        latitude=mask_lat,
                        longitude=mask_lon,
                        method="linear",
                    )

                    for mask_var in mask_variables:
                        mask = mask_ds.variables[mask_var][:]

                        # Calculate area-weighted mean precipitation.
                        # Convert rate per sec to total by multiplying by seconds and  number of days convert the CFS value
                        # to an estimated monthly total.
                        total_pcp = np.sum(pcp_remap * mask * area) * 60 * 60 * 24 * num_days
                        pcp_mm = total_pcp / np.sum(mask * area)

                        lake_abv, surface_type = mask_var.split("_")
                        lake = lake_lookup.get(lake_abv)
                        print(lake, pcp_mm.item())

                        if lake is None:
                            raise ValueError(
                                "ERROR: The mask variables need to begin with "
                                "'eri', 'ont', 'sup', or 'mih'. Check the mask file."
                            )

                        # Insert precipitation into the database.
                        self.db.add(
                            cfs_run,
                            forecast_year,
                            forecast_month,
                            lake,
                            surface_type,
                            "precipitation",
                            pcp_mm.item(),
                        )

                except Exception as e:
                    print(f"ERROR processing evaporation data: {e}. Skipping forecast.")
                    continue

            # -------------------------
            # Skip files that do not match expected CFS GRIB patterns
            # -------------------------
            else:
                print(f"Skipping unrecognized file: {filename}")
                continue

for date in date_array[:-1]:
    print(f"Beginning Files for {date}.")

    date_str = date.strftime('%Y%m%d%H')
    YYYY, MM, DD, HH = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d'), date.strftime('%H')

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



