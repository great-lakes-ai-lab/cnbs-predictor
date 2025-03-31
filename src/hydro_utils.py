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

def convert_cms_to_mm(df_cms):
    """
    Convert data from cubic meters per second (cms) to millimeters (mm) based on surface area and seconds in each month.
    
    Parameters:
    df_cms (pandas.DataFrame): A DataFrame containing data in [cms]. Each column should represent a different lake 
                               beginning with 'erie', 'superior', 'michigan-huron', 'ontario' and the index should be the year and month.
    
    Returns:
    pandas.DataFrame: Converted data in millimeters (mm), with the same structure as df_cms.
    
    Raises:
    ValueError: If the input DataFrame does not contain the expected columns or the surface area values are missing,
                or if the index does not contain valid year and month.
    """
    # Check if input is a DataFrame
    if not isinstance(df_cms, pd.DataFrame):
        raise ValueError("ERROR: Input must be a pandas DataFrame.")
    
    # Check if the required region columns are present in the DataFrame
    required_columns = ['erie', 'superior', 'michigan-huron', 'ontario']
    if not any(df_cms.columns.str.startswith(col) for col in required_columns):
        raise ValueError("ERROR: DataFrame must contain columns with region data starting with 'erie', 'superior', 'michigan-huron', 'ontario'.")
    
    # Check if the index contains year and month (index should be 2D: year and month)
    for idx in df_cms.index:
        if len(idx) != 2:
            raise ValueError(f"ERROR: Index should contain both year and month.")
        year, month = idx
        # Check that the year is an integer and the month is between 1 and 12
        if not isinstance(year, int) or not (1 <= month <= 12):
            raise ValueError(f"ERROR: Year should be an integer and month should be between 1 and 12.")
        
    # Create a copy of the dataframe so we aren't altering the df_cms
    df_mm = df_cms.copy()

    sa_sup = 82097*1000000
    sa_mih = (57753 + 5956)*1000000
    sa_eri = 25655*1000000
    sa_ont = 19009*1000000
    
    # Calculate the number of seconds for each month
    df_mm['seconds'] = df_mm.apply(lambda row: seconds_in_month(int(row.name[0]), int(row.name[1])), axis=1)

    # value_cms / surface_area * seconds_in_a_month = m * 1000 = mm
    for column in df_mm.columns:
        if column.startswith("erie"):
            df_mm[column] = df_mm[column] / sa_eri * df_mm['seconds'] * 1000
        elif column.startswith("superior"):
            df_mm[column] = df_mm[column] / sa_sup * df_mm['seconds'] * 1000
        elif column.startswith("michigan-huron"):
            df_mm[column] = df_mm[column] / sa_mih * df_mm['seconds'] * 1000
        elif column.startswith("ontario"):
            df_mm[column] = df_mm[column] / sa_ont * df_mm['seconds'] * 1000

    # Deleting column 'seconds'
    df_final_mm = df_mm.drop('seconds', axis=1)

    return df_final_mm

def convert_mm_to_cms(df_mm):
    """
    Convert data from millimeters (mm) to cubic meters per second (cms) based on surface area and seconds in each month.
    
    Parameters:
    df_mm (pandas.DataFrame): A DataFrame containing data in [mm]. Each column should represent a different lake 
                               beginning with 'erie', 'superior', 'michigan-huron', 'ontario' and the index should be the year and month.
    
    Returns:
    pandas.DataFrame: Converted data in cubic meters per second (cms), with the same structure as df_mm.
    
    Raises:
    ValueError: If the input DataFrame does not contain the expected columns or the surface area values are missing,
                or if the index does not contain valid year and month.
    """
# Check if input is a DataFrame
    if not isinstance(df_mm, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Check if the index contains year and month (index should be 2D: year and month)
    for idx in df_mm.index:
        if len(idx) != 2:
            raise ValueError(f"Index should contain both year and month, but index {idx} is invalid.")
        year, month = idx
        # Check that the year is an integer and the month is between 1 and 12
        if not isinstance(year, int) or not (1 <= month <= 12):
            raise ValueError(f"Invalid year or month in index {idx}. Year should be an integer and month should be between 1 and 12.")
    
    # Create a copy of the dataframe so we aren't altering the original df_mm
    df_cms = df_mm.copy()

    # Surface areas in square meters (multiplied by 10^6 to avoid working with large numbers)
    sa_sup = 82097 * 1000000
    sa_mih = (57753 + 5956) * 1000000
    sa_eri = 25655 * 1000000
    sa_ont = 19009 * 1000000
    
    # Calculate the number of seconds for each month
    # Assuming the `seconds_in_month` function is correctly defined
    df_cms['seconds'] = df_cms.apply(lambda row: seconds_in_month(int(row.name[0]), int(row.name[1])), axis=1)

    # Value conversion: value_mm / 1000 [to convert to m] * surface_area / seconds_in_a_month
    for column in df_mm.columns:
        if column.startswith("erie"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_eri) / df_cms['seconds']
        elif column.startswith("superior"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_sup) / df_cms['seconds']
        elif column.startswith("michigan-huron"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_mih) / df_cms['seconds']
        elif column.startswith("ontario"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_ont) / df_cms['seconds']

    # Deleting the 'seconds' column after calculation
    df_final_cms = df_cms.drop('seconds', axis=1)

    return df_final_cms

