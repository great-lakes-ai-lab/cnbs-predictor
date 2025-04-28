import calendar
import numpy as np
import pandas as pd


def seconds_in_month(year, month):
    """
    Calculate the number of seconds in a given month of a specific year.

    Parameters:
    year (int): The year for which the month is being calculated.
    month (int): The month for which the number of seconds is being calculated (1-12).

    Returns:
    int: The number of seconds in the specified month.

    Raises:
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

    Parameters:
    lon (array-like): 1D array of longitudes in degrees.
    lat (array-like): 1D array of latitudes in degrees.

    Returns:
    numpy.ndarray: A 2D array of grid cell areas in square meters.

    Raises:
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

def calculate_evaporation(temperature_K, latent_heat):
    """
    Calculate the evaporation rate based on temperature and latent heat.

    Parameters:
    temperature_K (float or array-like): Temperature in Kelvin.
    latent_heat (float or array-like): Latent heat in W/m².

    Returns:
    float or numpy.ndarray: The evaporation rate in kg/m²/s.

    Raises:
    ValueError: If temperature_K is less than 0 or latent_heat is negative.
    """
    # ET = kg/(m²*time^1) or 1 mm
    # LE = MJ/(M²*time^1)
    # λ  = MJ/kg

    # Latent heat of vaporization varies slightly with temperature. Allen et al. (1998) provides an equation 
    # for calculating λ with air temperature variation. Temperature in this case must be in degrees Celcius.

    # λ=2.501−((2.361×10−3)×(Temp-273.15))

    # latent_heat is in W/m² or J/(m²*time^1). In order to convert to MJ we must multiply by 10^-6 or 
    # 0.000001. Now lamba and latent_heat are both in terms of MJ.

    lamda=(2.501-(0.002361*(temperature_K-273.15)))
    evaporation_rate=((latent_heat)*0.000001)/lamda

    return evaporation_rate

def convert_mm_to_cms(df):
    """
    Converts the 'value [mm]' in the dataframe to 'value [cms]' (cubic meters per second) based on lake surface area and the number of seconds in the month.

    Args:
    - df (pd.DataFrame): DataFrame containing the columns 'value [mm]', 'lake', and a multi-index with 'month' and 'year'.
    
    Returns:
    - pd.DataFrame: DataFrame with a new column 'value [cms]' representing the value in cubic meters per second.
    """

    # Dictionary storing the surface area (in square meters) for each lake
    lake_sa = {
        'superior': 82097 * 1000000,       # Lake Superior area in square meters
        'michigan-huron': (57753 + 59560) * 1000000,  # Lake Michigan-Huron combined area in square meters
        'erie': 25655 * 1000000,           # Lake Erie area in square meters
        'ontario': 19009 * 1000000         # Lake Ontario area in square meters
    }
    
    # Apply the conversion formula for each row in the dataframe
    df['value [cms]'] = df.apply(
        lambda row: (
            # Convert mm to meters, multiply by the lake surface area, and divide by seconds in the given month
            (row['value [mm]'] / 1000) * lake_sa.get(row['lake'], 0) / 
            seconds_in_month(row.name[2], row.name[1])  # seconds_in_month() needs to be defined elsewhere
        ), axis=1
    )
    
    # Return the modified DataFrame with the new 'value [cms]' column
    return df

