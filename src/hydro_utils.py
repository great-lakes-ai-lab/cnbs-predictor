"""Hydrological and unit-conversion helpers for the forecast pipeline.

Small, mostly stateless functions used throughout data processing and
forecasting:

- time helpers (:func:`seconds_in_month`)
- grid geometry (:func:`calculate_grid_cell_areas`)
- physical conversions (:func:`calculate_evaporation_rate`,
  :func:`convert_mm_to_cms`)
- model loading (:func:`load_model`)

Lake surface areas and other constants are defined inline where used.
"""

import calendar
import numpy as np
import pandas as pd
import joblib


def seconds_in_month(year, month):
    """
    Calculate the number of seconds in a given month of a specific year.

    Parameters
    ----------
    year (int): The year for which the month is being calculated.
    month (int): The month for which the number of seconds is being calculated (1-12).

    Returns
    ----------
    int: The number of seconds in the specified month.

    Raises
    ----------
    ValueError: If the month is not between 1 and 12.
    """
    # Check if the month is between 1 and 12
    if month < 1 or month > 12:
        raise ValueError("ERROR: Month must be between 1 and 12.")
    
    # Get the number of days in the specified month and year
    num_days = calendar.monthrange(year, month)[1]
    
    # Convert the number of days to seconds (days * 24 hours * 60 minutes * 60 seconds)
    return num_days * 24 * 60 * 60

def calculate_grid_cell_areas(lon, lat):
    """
    Calculate the area of each grid cell given latitude and longitude arrays.

    Parameters
    ----------
    lon (array-like): 1D array of longitudes in degrees.
    lat (array-like): 1D array of latitudes in degrees.

    Returns
    ----------
    numpy.ndarray: A 2D array of grid cell areas in square meters.

    Raises
    ----------
    ValueError: If lat or lon are not 1D arrays.
    """
    
    # Check that lon and lat are 1D arrays
    if len(lon.shape) != 1 or len(lat.shape) != 1:
        raise ValueError("ERROR: Both 'lon' and 'lat' must be 1D arrays.")
    
    # Radius of Earth in meters
    R = 6371000.0
    
    # Convert latitude to radians
    lat_rad = np.radians(lat)
    
    # Calculate grid cell width in radians
    dlat = np.radians(lat[1] - lat[0])
    dlon = np.radians(lon[1] - lon[0])
    
    # Initialize the area array
    area = np.zeros((len(lat), len(lon)))
    
    # Calculate area of each grid cell in square meters
    for i in range(len(lat)):
        for j in range(len(lon)):
            area[i, j] = R**2 * dlat * dlon * np.cos(lat_rad[i])
    
    return area

def calculate_evaporation_rate(temperature_K, latent_heat_flux):
    """
    Convert latent heat flux to evaporation rate.

    Parameters
    ----------
    temperature_K : float or array
        Air temperature in Kelvin.

    latent_heat_flux_W_m2 : float or array
        Latent heat flux in W/m².

    Notes:
    No input validation is performed. Negative ``temperature_K`` or
    ``latent_heat`` values will produce numerically valid but physically
    meaningless results — callers are responsible for sanity-checking inputs.
    Sub-optimal: a future revision should raise ``ValueError`` for inputs
    outside physical ranges.
    
    Returns
    -------
    evaporation_rate_mm_s : float or array
        Evaporation rate in mm/s.
    """

    # Convert temperature to Celsius
    temperature_C = temperature_K - 273.15

    # Latent heat of vaporization (MJ/kg)
    lambda_MJ_kg = 2.501 - 0.002361 * temperature_C

    # Convert W/m² -> MJ/(s·m²)
    latent_heat_flux_MJ = latent_heat_flux * 1e-6

    # MJ/(s·m²) / MJ/kg = kg/(m²·s) = mm/s
    evaporation_rate_mm_s = latent_heat_flux_MJ / lambda_MJ_kg

    return evaporation_rate_mm_s

def convert_mm_to_cms(df):
    """
    Converts the 'value [mm]' in the dataframe to 'value [cms]' (cubic meters per second) based on lake surface area and the number of seconds in the month.

    Parameters
    ----------
    - df (pd.DataFrame): DataFrame containing the columns 'value [mm]', 'lake', and a multi-index with 'month' and 'year'.
    
    Returns
    ----------
    - pd.DataFrame: DataFrame with a new column 'value [cms]' representing the value in cubic meters per second.

    Notes:
    Recognized lake names are 'superior', 'michigan-huron', 'erie', and
    'ontario'. Any other value in the 'lake' column silently yields a
    'value [cms]' of 0 (surface area defaults to 0 via ``dict.get``).
    Sub-optimal: a future revision should raise on unknown lake names
    rather than masking the issue with zeros.
    """

    # Dictionary storing the surface area (in square meters) for each lake
    lake_sa = {
        'superior': 82097 * 1000000,       # Lake Superior area in square meters
        'michigan-huron': (57753 + 59560) * 1000000,  # Lake Michigan-Huron combined area in square meters
        'erie': 25655 * 1000000,           # Lake Erie area in square meters
        'ontario': 19009 * 1000000         # Lake Ontario area in square meters
    }
    df_units = df.copy()  # Create a copy of the input DataFrame to avoid modifying the original

    # Apply the conversion formula for each row in the dataframe
    df_units['value [cms]'] = df_units.apply(
        lambda row: (
            # Convert mm to meters, multiply by the lake surface area, and divide by seconds in the given month
            (row['value [mm]'] / 1000) * lake_sa.get(row['lake'], 0) / 
            seconds_in_month(row['year'], row['month'])
        ), axis=1
    )
    
    # Return the modified DataFrame with the new 'value [cms]' column
    return df_units

def load_model(model_name: str, models_info: list):
    """
    Load a serialized model by its name.

    Parameters
    ----------
    model_name : str
        The short identifier of the model (e.g., "GP", "RF").
    models_info : list of dict
        Each dict must contain:
        - "model": the short identifier
        - "path" : the path to the saved model file

    Returns
    -------
    object
        The deserialized model object.

    Raises
    ------
    ValueError
        If `model_name` isn’t found in `models_info`.
    FileNotFoundError
        If the `.joblib` file cannot be located.
    """
    try:
        path = next(m["path"] for m in models_info if m["model"] == model_name)
    except StopIteration:
        raise ValueError(f"Model '{model_name}' not found in models_info")

    return joblib.load(path)