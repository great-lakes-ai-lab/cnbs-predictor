import pandas as pd
import numpy as np
import calendar
import os

# Surface area [m²] for each lake
lake_areas = {
    'superior': 78288645587.81192,
    'michigan-huron': 123626283030.46616,
    'erie': 18596386416.712486,
    'ontario': 15569248531.837788
}

class DataLoader:
    def __init__(self):
        pass  # Add config or default paths here if needed

    def glcc(self, directory, units='cms'):
        """
        Load observed GLCC NBS data for all four lakes in either 'cms' or 'mm' units.
        """
        filenames = {
            'superior': 'LakeSuperior_MonthlyNetBasinSupply_1900to2025.csv',
            'michigan-huron': 'LakeMichiganHuron_MonthlyNetBasinSupply_1900to2025.csv',
            'erie': 'LakeErie_MonthlyNetBasinSupply_1900to2025.csv',
            'ontario': 'LakeOntario_MonthlyNetBasinSupply_1900to2025.csv',
        }

        def format_data(df):
            df_long = df.melt(id_vars=['Year'], var_name='Month', value_name='Observed')
            df_long['date'] = pd.to_datetime(
                df_long['Year'].astype(str) + '-' + df_long['Month'].str[1:4] + '-01',
                format='%Y-%b-%d'
            )
            return df_long[['date', 'Observed']].sort_values('date').reset_index(drop=True)

        dfs = []
        for lake, filename in filenames.items():
            path = os.path.join(directory, filename)
            df = pd.read_csv(path, skiprows=11)
            formatted = format_data(df).rename(columns={'Observed': f'{lake}_nbs_obs'})
            dfs.append(formatted)

        # Merge all dataframes on 'date'
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge(df_merged, df, on='date', how='outer')

        df_merged.replace(-99990.0, np.nan, inplace=True)

        lake_cols = [col for col in df_merged.columns if col.endswith('_nbs_obs')]
        df_merged.dropna(subset=lake_cols, how='all', inplace=True)

        if units == 'mm':

            df_merged['seconds_in_month'] = df_merged['date'].apply(
                lambda x: calendar.monthrange(x.year, x.month)[1] * 24 * 60 * 60
            )

            for col in lake_cols:
                lake = col.replace('_nbs_obs', '')
                area = lake_areas.get(lake)
                if area:
                    df_merged[col] = (
                        df_merged[col] * df_merged['seconds_in_month'] / area * 1000
                    )

            df_merged.drop(columns='seconds_in_month', inplace=True)

        df_merged.set_index('date', inplace=True)
        return df_merged

    def l2swbm(self, directory):
        """
        Load L2SWBM runoff, evaporation, and precipitation for all lakes.
        """
        lakes = {
            'superior': 'sup',
            'michigan-huron': 'mih',
            'erie': 'eri',
            'ontario': 'ont'
        }
        variables = ['Evap', 'Runoff', 'Precip']
        dfs = []

        for lake_name in lakes.keys():
            lake_data = {}
            for var in variables:
                lake_file_name = lake_name.replace('michigan-huron', 'miHuron')
                file_path = os.path.join(directory, f'{lake_file_name}{var}_MonthlyRun.csv')

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

        final_df.set_index('date', inplace=True)
        final_df.rename(columns=lambda col: col.replace('precip_', 'precipitation_').replace('evap_', 'evaporation_'), inplace=True)
        return final_df

    def glsea(self, file_path, units='K'):
        """
        Reads GLSEA SST data from a file and returns a cleaned DataFrame.
        """
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=6, comment='-')

        df['date'] = pd.to_datetime(df['Year'].astype(str) + df['Day'].astype(str), format='%Y%j')
        df.set_index('date', inplace=True)
        df.drop(columns=['Year', 'Day'], inplace=True)

        df.columns = ['superior_sst', 'michigan_sst', 'huron_sst', 'erie_sst', 'ontario_sst', 'stclair_sst']
        df['michigan-huron_sst'] = (df['michigan_sst'] + df['huron_sst']) / 2
        df = df[['superior_sst', 'michigan-huron_sst', 'erie_sst', 'ontario_sst']]

        if units.upper() == 'K':
            df += 273.15
        elif units.upper() != 'C':
            raise ValueError("Unsupported units. Use 'C' for Celsius or 'K' for Kelvin.")

        return df

    def lake_probabilities(self, file_dir, units = "cms"):
        """
        Load and merge probability data for the Great Lakes.

        Parameters
        ----------
        file_dir : str
            Directory containing the probability CSV files (e.g., '.../data/probabilities/').
        units : str, optional
            Units for output values. 
            Options:
            - 'cms' (default): keep values as discharge (m³/s)
            - 'mm' : convert to lake-equivalent depth (mm/month)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['month', 'superior', 'michigan-huron', 'erie', 'ontario'].
        """

        # Lake files
        lake_files = {
            "superior": "SUP.probs.csv",
            "michigan-huron": "MIH.probs.csv",
            "erie": "ERI.probs.csv",
            "ontario": "ONT.probs.csv",
        }

        # Days in each month (non-leap year)
        days_in_month = {
            "Jan": 31, "Feb": 28, "Mar": 31, "Apr": 30, "May": 31, "Jun": 30,
            "Jul": 31, "Aug": 31, "Sep": 30, "Oct": 31, "Nov": 30, "Dec": 31
        }

        # --- Internal helper to process each lake file ---
        def process_lake(filepath, lake_name):
            df = pd.read_csv(filepath, skiprows=7)
            filtered = df[df["Probability Of Exceedance"] == 0.5]

            lake_df = (
                filtered.drop(columns=["Probability Of Exceedance"])
                .T
                .rename(columns={filtered.index[0]: lake_name})
                .reset_index()
                .rename(columns={"index": "month"})
            )

            # Optional unit conversion: m³/s → mm/month
            if units.lower() == "mm":
                area = lake_areas[lake_name]
                # Conversion factor: m³/s → mm/month
                # 1 m³/s = (1 / area) m/s over lake surface
                # Convert m → mm (×1000) and seconds → days × 86400
                lake_df[lake_name] = lake_df.apply(
                    lambda row: row[lake_name] * 1000 * 86400 * days_in_month.get(row["month"], 30) / area,
                    axis=1
                )

            return lake_df

        # --- Process and merge all ---
        merged_df = None
        for lake, filename in lake_files.items():
            lake_df = process_lake(file_dir + filename, lake)
            if merged_df is None:
                merged_df = lake_df
            else:
                merged_df = merged_df.merge(lake_df, on="month")

        return merged_df