# This script was used on Umich's Great Lakes cluster to download the CFS reforecast data from NOAA's NCEI.
# Lindsay Fitzpatrick
# 2026-07-16

import os
import requests
import pandas as pd

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
start_date = "1981-12-12"
end_date   = "2011-04-01"

forecast_hours = ["00", "06", "12", "18"]
n_months = 11

# Download files
# We only need to download the flux files. 'tp' in pgb is 0 throughout, so we use 'prate'
# in the flux files to get the precipitation rate. The pgb files are also much larger than the flux files so this works to
# reduce the amount of data we need to download.
#file_prefixes = ["pgbf", "flxf"]
file_prefixes = ["flxf"]

BASE_S3_URL = (
    "https://www.ncei.noaa.gov/oa/prod-cfs-reforecast/"
    "high-priority-subset/monthly-means-9-month/"
)

local_dir = "/scratch/dannes_root/dannes0/ljob/cfs_reforecast/"

# Every day (only every 5th will actually exist)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

session = requests.Session()

# -----------------------------------------------------------------------------
# Loop over initialization dates
# -----------------------------------------------------------------------------
for date in dates:

    year = date.year
    month = date.month
    init_ymd = date.strftime("%Y%m%d")

    print(f"\nChecking {init_ymd}")

    PREFIX_ROOT = f"{year}/{year}{month:02d}/{init_ymd}/"

    # -------------------------------------------------------------------------
    # Check if this initialization date exists by testing one known file
    # -------------------------------------------------------------------------
    test_valid = date.strftime("%Y%m")

    test_file = (
        f"pgbf"
        f"{init_ymd}00.01."
        f"{test_valid}.avrg.grb2"
    )

    test_url = f"{BASE_S3_URL}{PREFIX_ROOT}{test_file}"

    try:
        response = session.head(test_url, timeout=20)

        if response.status_code == 404:
            print("No files for this initialization date.")
            continue

        response.raise_for_status()

    except requests.RequestException:
        print("No files for this initialization date.")
        continue

    # -------------------------------------------------------------------------
    # Create local directory ONLY if files exist
    # -------------------------------------------------------------------------
    date_dir = os.path.join(local_dir, init_ymd)
    os.makedirs(date_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Download all files
    # -------------------------------------------------------------------------
    for prefix in file_prefixes:

        for fh in forecast_hours:

            for lead in range(n_months):

                valid_ym = (
                    date + pd.DateOffset(months=lead)
                ).strftime("%Y%m")

                fname = (
                    f"{prefix}"
                    f"{init_ymd}{fh}.01."
                    f"{valid_ym}.avrg.grb2"
                )

                file_url = f"{BASE_S3_URL}{PREFIX_ROOT}{fname}"
                out_path = os.path.join(date_dir, fname)

                if os.path.exists(out_path):
                    continue

                try:
                    r = session.get(file_url, stream=True, timeout=60)

                    if r.status_code == 404:
                        continue

                    r.raise_for_status()

                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    print(f"Downloaded {fname}")

                except requests.RequestException as e:
                    print(f"Failed: {fname}")
                    print(e)

    print(f"Finished {init_ymd}")
