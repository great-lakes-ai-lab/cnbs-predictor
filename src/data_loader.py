import pandas as pd
import numpy as np
import calendar
import os
import requests
from io import StringIO

def load_glcc_data(directory, units):
    """
    Load and return observed GLCC NBS data for all four lakes in either 'cms' or 'mm' units.
    
    Parameters:
        directory (str): Path to the directory where GLCC CSV files are stored.
        units (str): 'cms' for cubic meters per second (default in files), or 'mm' for millimeters per month.

    Returns:
        pd.DataFrame: Merged DataFrame with date and observed values for each lake.
    """
    # Load raw GLCC data
    observed_sup = pd.read_csv(directory + 'LakeSuperior_MonthlyNetBasinSupply_1900to2025.csv', skiprows=11)
    observed_mih = pd.read_csv(directory + 'LakeMichiganHuron_MonthlyNetBasinSupply_1900to2025.csv', skiprows=11)
    observed_eri = pd.read_csv(directory + 'LakeErie_MonthlyNetBasinSupply_1900to2025.csv', skiprows=11)
    observed_ont = pd.read_csv(directory + 'LakeOntario_MonthlyNetBasinSupply_1900to2025.csv', skiprows=11)

    def format_observed_data(df):
        """Convert wide GLCC observed data into long format with year-month datetime."""
        observed_df = df.melt(id_vars=['Year'], var_name='Month', value_name='Observed')
        observed_df['date'] = pd.to_datetime(observed_df['Year'].astype(str) + '-' + observed_df['Month'].str[1:4] + '-01',
                              format='%Y-%b-%d')
        observed_df = observed_df[['date', 'Observed']].sort_values(by='date').reset_index(drop=True)
        return observed_df

    # Format and rename
    df_obs_sup = format_observed_data(observed_sup).rename(columns={'Observed': 'superior_nbs_obs'})
    df_obs_mih = format_observed_data(observed_mih).rename(columns={'Observed': 'michigan-huron_nbs_obs'})
    df_obs_eri = format_observed_data(observed_eri).rename(columns={'Observed': 'erie_nbs_obs'})
    df_obs_ont = format_observed_data(observed_ont).rename(columns={'Observed': 'ontario_nbs_obs'})

    # Merge all into one dataframe
    df_obs_merged = pd.merge(df_obs_sup, df_obs_mih, on="date", how="outer")
    df_obs_merged = pd.merge(df_obs_merged, df_obs_eri, on="date", how="outer")
    df_obs_merged = pd.merge(df_obs_merged, df_obs_ont, on="date", how="outer")

    # Replace -99990.0 with NaN
    df_obs_merged.replace(-99990.0, np.nan, inplace=True)

    # Drop rows where all lake observations are NaN
    lake_cols = [col for col in df_obs_merged.columns if col.endswith('_nbs_obs')]
    df_obs_merged.dropna(subset=lake_cols, how='all', inplace=True)

    # Convert to millimeters if requested
    if units == 'mm':
        # Surface area [m²] for each lake
        lake_areas = {
            'superior': 78288645587.81192,
            'michigan-huron': 123626283030.46616,
            'erie': 18596386416.712486,
            'ontario': 15569248531.837788
        }

        df_obs_merged['seconds_in_month'] = df_obs_merged['date'].apply(
            lambda x: calendar.monthrange(x.year, x.month)[1] * 24 * 60 * 60
        )

        for col in lake_cols:
            lake = col.replace('_nbs_obs', '')
            area = lake_areas.get(lake)
            if area:
                df_obs_merged[col] = (
                    df_obs_merged[col] * df_obs_merged['seconds_in_month'] / area * 1000
                )

        df_obs_merged.drop(columns='seconds_in_month', inplace=True)

    df_obs_merged.set_index('date', inplace=True)

    return df_obs_merged

def load_l2swbm_data(folder):
    lakes = {
        'superior': 'sup',
        'michigan-huron': 'mih',
        'erie': 'eri',
        'ontario': 'ont'
    }
    variables = ['Evap', 'Runoff', 'Precip']

    dfs = []

    for lake_name, short in lakes.items():
        lake_data = {}
        for var in variables:
            # Replace dash with nothing in filename since files use "miHuron" not "michigan-huron"
            lake_file_name = lake_name.replace('michigan-huron', 'miHuron')
            file_path = os.path.join(folder, f'{lake_file_name}{var}_MonthlyRun.csv')

            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(
                df['Year'].astype(int).astype(str) + '-' +
                df['Month'].astype(int).astype(str).str.zfill(2)
            )
            col_name = f'{lake_name.lower()}_{var.lower()}_obs'
            lake_data[col_name] = df['Median']
            if 'date' not in lake_data:
                lake_data['date'] = df['date']
        lake_df = pd.DataFrame(lake_data)
        dfs.append(lake_df)

    # Merge all lake dataframes on 'date'
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on='date')

    # Set 'date' as index
    final_df.set_index('date', inplace=True)
    final_df.rename(columns=lambda col: col.replace('precip_', 'precipitation_').replace('evap_', 'evaporation_'), inplace=True)

    return final_df

def load_glsea_data(file_path, units='K'):
    """
    Reads GLSEA SST data from a file and returns a cleaned DataFrame with selected columns.
    
    Parameters:
        file_path (str): Path to the SST CSV file.
        units (str): Temperature units, either 'K' (Kelvin) or 'C' (Celsius). Default is 'K'.
    
    Returns:
        pd.DataFrame: Processed SST DataFrame with selected lakes and optional unit conversion.
    """
    # Read the data
    df_sst = pd.read_csv(file_path, sep=r'\s+', skiprows=6, comment='-')

    # Convert Year and Day-of-year into datetime
    df_sst['date'] = pd.to_datetime(df_sst['Year'].astype(str) + df_sst['Day'].astype(str), format='%Y%j')
    df_sst.set_index('date', inplace=True)
    df_sst.drop(columns=['Year', 'Day'], inplace=True)

    # Rename columns
    df_sst.columns = ['superior_sst', 'michigan_sst', 'huron_sst', 'erie_sst', 'ontario_sst', 'stclair_sst']

    # Michigan-Huron average
    df_sst['michigan-huron_sst'] = (df_sst['michigan_sst'] + df_sst['huron_sst']) / 2

    # Select columns
    df_sst = df_sst[['superior_sst', 'michigan-huron_sst', 'erie_sst', 'ontario_sst']]

    # Convert units if required
    if units.upper() == 'K':
        df_sst += 273.15
    elif units.upper() != 'C':
        raise ValueError("Unsupported units. Use 'C' for Celsius or 'K' for Kelvin.")

    return df_sst

def get_current_ssts(url):
    """
    Fetch the full daily lake average surface water temperature
    from a NOAA GLSEA .dat file.

    Parameters
    ----------
    url : str
        The URL of the .dat file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Date, Superior, Michigan-Huron, Erie, Ontario
        or None if the URL is not accessible.
    """

    # Download file as text
    response = requests.get(url)
    response.raise_for_status()
    text_data = response.text

    # Keep only data lines (start with a digit)
    lines = text_data.strip().split("\n")
    data_lines = [line for line in lines if line.strip() and line[0].isdigit()]

    # Column names from file
    col_names = ["year", "day", "superior_sst", "michigan_sst", "huron_sst", "erie_sst", "ontario_sst", "st.clair_sst"]

    # Read all data lines into DataFrame
    df = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", names=col_names)

    # Convert Year + Julian Day → datetime
    df["date"] = pd.to_datetime(df["year"].astype(str) + df["day"].astype(str), format="%Y%j")

    # Compute Michigan-Huron average
    df["michigan-huron_sst"] = df[["michigan_sst", "huron_sst"]].mean(axis=1)

    # Select and rename desired columns
    df = df[["date", "superior_sst", "michigan-huron_sst", "erie_sst", "ontario_sst"]]

    # Convert temperatures from Celsius to Kelvin
    for col in ["superior_sst", "michigan-huron_sst", "erie_sst", "ontario_sst"]:
        df[col] = df[col] + 273.15

    df.set_index('date', inplace=True)

    return df