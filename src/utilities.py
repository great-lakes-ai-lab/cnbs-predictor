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
        List files in a directory whose names match a given prefix or suffix.

        Parameters
        ----------
        directory : str
            Path to the directory to search.
        affix : str
            Which part of the filename to match against: ``'prefix'`` or
            ``'suffix'``.
        identifier : str
            The prefix or suffix string to match.

        Returns
        -------
        list of str
            Full paths (``directory`` joined with the filename) of all matching
            files. Empty if nothing matches.
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

def format_cep_mean_forecast(
    df_cep,
    output_file=None,
    models=None
):
    """
    Format CEP forecast dataframe for reporting.

    Parameters
    ----------
    df_cep : pandas.DataFrame
        Long-format CEP forecast dataframe containing:
        'forecast_month', 'model', 'lake', and 'cep'.

    output_file : str, optional
        Path to save formatted CSV. If None, file is not saved.

    models : list, optional
        Ordered list of models to include.

    Returns
    -------
    pandas.DataFrame
        Formatted CEP mean dataframe.
    """

    if models is None:
        models = ["RF", "GP", "XGB", "NN"]

    # ---------------------------------------------------------
    # Start with copy
    # ---------------------------------------------------------
    df_format = df_cep.copy()

    # Ensure forecast_month is datetime
    df_format["forecast_month"] = pd.to_datetime(
        df_format["forecast_month"],
        format="%Y-%m"
    )

    # Lake abbreviations
    lake_map = {
        "superior": "SUP",
        "michigan-huron": "MIH",
        "erie": "ERI",
        "ontario": "ONT"
    }

    df_format["lake"] = df_format["lake"].map(lake_map)

    # Extract year and month
    df_format["year"] = df_format["forecast_month"].dt.year
    df_format["month"] = df_format["forecast_month"].dt.strftime("%b")

    # ---------------------------------------------------------
    # Pivot models into columns
    # ---------------------------------------------------------
    df_format["model"] = pd.Categorical(
        df_format["model"],
        categories=models,
        ordered=True
    )

    df_format = df_format.sort_values(
        ["forecast_month", "model", "lake"]
    ).reset_index(drop=True)

    df_final = df_format.pivot_table(
        index=["lake", "year", "month"],
        columns="model",
        values="cep"
    ).reset_index()

    df_final.columns.name = None

    # Ensemble mean
    df_final["MEAN"] = (
        df_final[models]
        .mean(axis=1)
        .round(2)
    )

    # ---------------------------------------------------------
    # Apply ordering
    # ---------------------------------------------------------
    lake_order = ["SUP", "MIH", "ERI", "ONT"]

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    df_final["lake"] = pd.Categorical(
        df_final["lake"],
        categories=lake_order,
        ordered=True
    )

    df_final["month"] = pd.Categorical(
        df_final["month"],
        categories=month_order,
        ordered=True
    )

    df_final = df_final.sort_values(
        ["lake", "year", "month"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Add blank rows between lakes
    # ---------------------------------------------------------
    df_final["lake"] = df_final["lake"].astype(str)

    formatted_groups = []

    for i, lake in enumerate(lake_order):

        group = df_final[df_final["lake"] == lake].copy()

        # Only display lake name on first row
        if len(group) > 1:
            group.loc[group.index[1:], "lake"] = ""

        formatted_groups.append(group)

        # Blank row between lakes
        if i < len(lake_order) - 1:
            formatted_groups.append(
                pd.DataFrame(
                    [[""] * len(df_final.columns)],
                    columns=df_final.columns
                )
            )

    df_final = pd.concat(
        formatted_groups,
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Format values
    # ---------------------------------------------------------
    value_cols = models + ["MEAN"]

    for col in value_cols:
        df_final[col] = df_final[col].map(
            lambda x: f"{x:.2f}" if pd.notna(x) and x != "" else ""
        )

    # Save
    if output_file is not None:
        df_final.to_csv(
            output_file,
            index=False,
            sep="\t"
        )

    return df_final