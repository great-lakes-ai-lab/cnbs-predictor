import calendar
import numpy as np

def seconds_in_month(year, month):
    # Number of days in the month
    num_days = calendar.monthrange(year, month)[1]
    # Convert days to seconds
    return num_days * 24 * 60 * 60

def calculate_grid_cell_areas(lon, lat):
    # Calculate grid cell areas
    # Assuming lat and lon are 1D arrays
    # Convert latitude to radians

    R = 6371000.0  # Radius of Earth in meters
    lat_rad = np.radians(lat)

    # Calculate grid cell width in radians
    dlat = np.radians(lat[1] - lat[0])
    dlon = np.radians(lon[1] - lon[0])

    # Calculate area of each grid cell in square kilometers
    area = np.zeros((len(lat), len(lon)))
    for i in range(len(lat)):
        for j in range(len(lon)):
            area[i, j] = R**2 * dlat * dlon * np.cos(lat_rad[i])

    return area

def calculate_evaporation(temperature_K, latent_heat):
    # ET = kg/(m^2*time^1) or 1 mm
    # LE = MJ/(M^2*time^1)
    # λ  = MJ/kg

    # Latent heat of vaporization varies slightly with temperature. Allen et al. (1998) provides an equation 
    # for calculating λ with air  temperature variation. Temperature in this case must be in degrees Celcius.

    # λ=2.501−(2.361×10−3)×Temp Celcius

    # so for our data with Temp in Kelvin...

    # λ=2.501−((2.361×10−3)×(Temp-273.15))

    # Our variable_lhf is in W/m^2 or J/(m^2*time^1). In order to convert to MJ we must multiply by 10^-6 or 
    # 0.000001. Now we have lamba and variable_lhf both in terms of MJ.

    # Equation below will provide an evaporation rate in kg/m2 per s.

    lamda=(2.501-(0.002361*(temperature_K-273.15)))
    evaporation_rate=((latent_heat)*0.000001)/lamda

    return evaporation_rate # kg/m2 per s

def convert_cms_to_mm(df_cms):

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
        if column.startswith("eri"):
            df_mm[column] = df_mm[column] / sa_eri * df_mm['seconds'] * 1000
        elif column.startswith("sup"):
            df_mm[column] = df_mm[column] / sa_sup * df_mm['seconds'] * 1000
        elif column.startswith("mih"):
            df_mm[column] = df_mm[column] / sa_mih * df_mm['seconds'] * 1000
        elif column.startswith("ont"):
            df_mm[column] = df_mm[column] / sa_ont * df_mm['seconds'] * 1000

    # Deleting column 'seconds'
    df_final_mm = df_mm.drop('seconds', axis=1)

    return df_final_mm

def convert_mm_to_cms(df_mm):

    df_cms = df_mm.copy()

    sa_sup = 82097*1000000
    sa_mih = (57753 + 5956)*1000000
    sa_eri = 25655*1000000
    sa_ont = 19009*1000000
    
    # Calculate the number of seconds for each month
    df_cms['seconds'] = df_cms.apply(lambda row: seconds_in_month(int(row.name[0]), int(row.name[1])), axis=1)

    # value_mm / 1000 [to convert to m] * surface_area / seconds_in_a_month
    for column in df_mm.columns:
        if column.startswith("eri"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_eri) / df_cms['seconds']
        elif column.startswith("sup"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_sup) / df_cms['seconds']
        elif column.startswith("mih"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_mih) / df_cms['seconds']
        elif column.startswith("ont"):
            df_cms[column] = ((df_cms[column] / 1000) * sa_ont) / df_cms['seconds']

    # Deleting column 'seconds'
    df_final_cms = df_cms.drop('seconds', axis=1)

    return df_final_cms

