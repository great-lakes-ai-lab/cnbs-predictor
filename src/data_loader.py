"""Loaders for observed and training datasets used by the forecast tool.

This module reads the various on-disk datasets the model is trained and
validated against, returning each as a tidy, date-indexed
:class:`pandas.DataFrame`:

- **GLCC** observed Net Basin Supply (:meth:`DataLoader.glcc`)
- **L2SWBM** precipitation, evaporation, and runoff (:meth:`DataLoader.l2swbm`)
- **GLSEA** lake surface temperatures (:meth:`DataLoader.glsea`)
- **SNODAS** snow water equivalent (:meth:`DataLoader.snodas`)
- USACE probability-of-exceedance tables (:meth:`DataLoader.lake_probabilities`)

Several loaders accept a ``units`` argument and convert between cubic metres
per second (``cms``) and millimetres per month (``mm``) using the per-lake
surface areas in :data:`lake_areas`.
"""

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
    """Load the observed and training datasets used to fit and validate models.

    Each method reads one dataset family from a local directory and returns a
    cleaned, date-indexed :class:`pandas.DataFrame`. The loader is stateless;
    all paths and options are passed per call.
    """

    def __init__(self):
        """Initialize the DataLoader. No configuration is required."""
        pass

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

    def snodas(self,file):
        """
        Load and process SNODAS SWE data.

        Parameters
        ----------
        local_path : str
            Base path to local data directory.

        X_df : pd.DataFrame, optional
            DataFrame to align SNODAS data with by index.

        Returns
        -------
        snodas : pd.DataFrame
            Processed SNODAS dataframe indexed by date.

        aligned_snodas : pd.DataFrame, optional
            SNODAS aligned to X_df index.

        aligned_X_df : pd.DataFrame, optional
            X_df aligned to SNODAS index.
        """

        # -----------------------------
        # Area weights
        # -----------------------------
        eri_lake = 18596.39
        eri_land = 81902.18

        ont_lake = 15569.25
        ont_land = 64465.19

        mih_lake = 123626.28
        mih_land = 236478.94

        sup_lake = 78288.65
        sup_land = 123736.05

        mic_land = 118000.00
        hur_land = 134100.00

        # -----------------------------
        # Load data
        # -----------------------------
        snodas_data = pd.read_csv(
            file,
            sep=',',
            skiprows=5
        )

        # -----------------------------
        # Create datetime column
        # -----------------------------
        snodas_data['date'] = pd.to_datetime(
            snodas_data[['year', 'month', 'day']]
        )

        # -----------------------------
        # Michigan-Huron weighted merge
        # -----------------------------
        snodas_data['michigan-huron_swe'] = (
            (snodas_data['MIC'] * mic_land) +
            (snodas_data['HUR'] * hur_land)
        ) / mih_land

        # -----------------------------
        # Rename columns
        # -----------------------------
        snodas_data = snodas_data.rename(columns={
            'SUP': 'superior_swe',
            'ERI': 'erie_swe',
            'ONT': 'ontario_swe'
        })

        # -----------------------------
        # Keep desired columns
        # -----------------------------
        snodas = snodas_data[[
            'date',
            'superior_swe',
            'michigan-huron_swe',
            'erie_swe',
            'ontario_swe'
        ]]

        # -----------------------------
        # Set index
        # -----------------------------
        snodas.set_index('date', inplace=True)

        return snodas

    def lake_probabilities(self, file_dir, units="cms"):
        """
        Reads probability of exceedance data for multiple Great Lakes from CSV files,
        converts units if requested, and combines them into a single tidy (long-format) DataFrame.

        Each input CSV contains probability of exceedance values (0.99–0.01)
        for 12 monthly columns (Jan–Dec). The function reshapes and merges all
        lakes into one DataFrame suitable for analysis or plotting.

        Parameters
        ----------
        file_dir : str
            Path to the directory containing the lake probability CSV files.
            Expected files are:

            - SUP.probs.csv  (Lake Superior)
            - MIH.probs.csv  (Lakes Michigan-Huron)
            - ERI.probs.csv  (Lake Erie)
            - ONT.probs.csv  (Lake Ontario)

        units : str, optional, default="cms"
            Desired output units for flow values.

            - "cms" : cubic meters per second (original units from source files)
            - "mm"  : millimeters per month, computed using lake surface area
              and number of days in each month.

        Returns
        -------
        pandas.DataFrame
            A tidy DataFrame containing columns:

            - "month" : str, calendar month (Jan–Dec)
            - "lake" : str, lake name ("superior", "michigan-huron", "erie", "ontario")
            - "prob_exceedance" : float, probability of exceedance (0.99–0.01)
            - "value" : float, flow or equivalent value in chosen units

        Notes
        -----
        - When units="mm", values are converted from m³/s to mm/month using
          ``value_mm = value_cms * 1000 * 86400 * days_in_month / lake_area``,
          where ``lake_area`` is the surface area of each lake in m².
        - Input CSVs are assumed to have 7 header rows before the data block,
          and contain a column named "Probability Of Exceedance".

        Example
        -------
        >>> df = combine_lake_probabilities("/path/to/files/", units="mm")
        >>> df.query("lake == 'erie' and month == 'Jan'")
        """

        # --- Lake file mapping and surface areas (m²) ---
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
            """
            Reads and reshapes one lake's probability of exceedance data.
            """
            # Read file, skipping header lines
            df = pd.read_csv(filepath, skiprows=7)

            # Extract probability levels and remove the column
            prob_values = df["Probability Of Exceedance"]
            df = df.drop(columns=["Probability Of Exceedance"])

            # Transpose: make months rows, probabilities columns
            df_t = df.T
            df_t.columns = prob_values
            df_t = df_t.reset_index().rename(columns={"index": "month"})

            # Optional conversion to mm/month
            if units.lower() == "mm":
                area = lake_areas[lake_name]
                for col in prob_values:
                    df_t[col] = df_t.apply(
                        lambda row: row[col] * 1000 * 86400 * days_in_month.get(row["month"], 30) / area,
                        axis=1
                    )

            # Reshape to long (tidy) format
            df_long = df_t.melt(id_vars=["month"], var_name="prob_exceedance", value_name="value")
            df_long["lake"] = lake_name
            return df_long

        # --- Process and combine all lakes ---
        all_lakes = []
        for lake, filename in lake_files.items():
            lake_df = process_lake(file_dir + filename, lake)
            all_lakes.append(lake_df)

        merged_df = pd.concat(all_lakes, ignore_index=True)

        # Clean and reorder
        merged_df["prob_exceedance"] = merged_df["prob_exceedance"].astype(float)
        merged_df = merged_df.sort_values(["lake", "prob_exceedance"]).reset_index(drop=True)
        merged_df = merged_df[["month", "lake", "prob_exceedance", "value"]]
        
        return merged_df
