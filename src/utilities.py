"""General-purpose helpers shared across the forecast tool.

A grab-bag of small utilities used by the downloaders, processors, and
notebooks:

- :func:`check_url_exists` — test whether a remote resource is reachable
- :func:`get_first_forecast_month` — derive the operational forecast month
  from a reference date
- :func:`get_date_range` — resolve the start/end dates for a CFS download,
  either automatically (from the database) or from manual input
- :func:`create_directory` — create a directory if it does not exist
- :func:`get_files` — list files in a directory matching a prefix or suffix
"""

import requests
from datetime import datetime
import pandas as pd
import os

def check_url_exists(url):
    """
    Check whether a URL exists and is reachable via a HEAD request.

    Parameters
    ----------
    url : str
        The full URL to be checked.

    Returns
    -------
    bool
        True if the URL returns status code 200, False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"URL check failed: {e}")
        return False

from datetime import datetime
from dateutil.relativedelta import relativedelta


def get_first_forecast_month(today=None):
    """
    Returns the first forecast month based on today's date,
    then shifts backward by a specified number of months.

    Rules
    -----
    - If today's day is before the 26th, use the current month.
    - If today's day is on or after the 26th, use the following month.
    - Then subtract `months_back` months.

    Parameters
    ----------
    today : datetime, optional
        Reference date. Defaults to current date if None.

    Returns
    -------
    str
        Forecast month in 'YYYY-MM' format.
    """

    if today is None:
        today = datetime.today()

    # Determine operational forecast month
    if today.day < 26:
        forecast_month = datetime(today.year, today.month, 1)
    else:
        forecast_month = (
            datetime(today.year, today.month, 1)
            + relativedelta(months=1)
        )

    formatted_month = forecast_month.strftime('%m-%Y')

    print(f"First forecast month: {formatted_month}")

    return formatted_month

def get_date_range(db=None, auto='yes', start_date=None, end_date=None):
    """
    Determine the start and end dates for CFS CSV downloads, and return the date range.

    Parameters
    ----------
    db : object, optional
        Database object with a `get_next_run()` method. Required if auto='yes'.
    auto : str, default 'yes'
        Whether to automatically fetch the next run date from the database ('yes' or 'no').
    start_date : str, optional
        Manual start date in format 'MM-DD-YYYY'. Required if auto='no'.
    end_date : str, optional
        Manual end date in format 'MM-DD-YYYY'. Required if auto='no'.

    Returns
    -------
    tuple
        (start_date: datetime, end_date: datetime, date_array: pd.DatetimeIndex)
    """
    if auto.lower() == 'yes':
        if db is None:
            raise ValueError("Database object must be provided when auto='yes'.")
        # Fetch next cfs_run date and use yesterday's date for the end
        start_date = db.get_next_run()
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    else:
        # Convert manual input strings to datetime
        start_date = datetime.strptime(start_date, "%m-%d-%Y")
        end_date = datetime.strptime(end_date, "%m-%d-%Y")

    # Validate dates
    if start_date == end_date:
        print("The CSV files are up-to-date.")
    elif start_date > end_date:
        raise ValueError("End date cannot be older than start date. Try again.")
    else:
        print(f"Starting from: {start_date.strftime('%m-%d-%Y')} and continuing through: {end_date.strftime('%m-%d-%Y')}")

    # Create a daily date range
    date_array = pd.date_range(start=start_date, end=end_date, freq='1d')
    
    return start_date, end_date, date_array

def create_directory(directory):
        """Create a directory if it doesn't already exist."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Directory '{directory}' created.")
            else:
                print(f"Directory '{directory}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create the directory '{directory}'.")
        except Exception as e:
            print(f"ERROR occurred while creating the directory '{directory}': {e}")

def get_files(directory, affix, identifier):
        """
        Get a list of all files in the specified directory that match the given prefix or suffix.

        Notes
        -----
        Sub-optimal: this function is missing a ``return files`` statement and
        therefore always implicitly returns ``None`` regardless of what it
        finds. Callers currently get nothing back. A future revision should
        add the return statement so the matched paths are actually exposed.
        """
        files = []
        for file_name in os.listdir(directory):
            if affix == 'suffix':
                if file_name.endswith(identifier):
                    files.append(os.path.join(directory, file_name))
            elif affix == 'prefix':
                if file_name.startswith(identifier):
                    files.append(os.path.join(directory, file_name))
        return files