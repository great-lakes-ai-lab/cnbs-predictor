import os
import pandas as pd
import cfgrib
import sqlite3
import numpy as np
import calendar
from datetime import datetime
import joblib
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.database_utils import CFSDatabase
from src.hydro_utils import calculate_evaporation, calculate_grid_cell_areas

class CFSProcessor:
    def __init__(self, database, table):
        """
        Initialize the processor with database path and table name.
        """
        self.database = database
        self.table = table
        self.db = CFSDatabase(database, table)

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

    def shift_variables(df, lag=0, lead=0):
        """
        Create variables columns to include lags and leads.
        """
        df = df.copy()
        new_columns = []
        for column in df.columns:
            for lag_month in range(1, lag):
                new_columns.append(df[column].shift(lag_month).rename(f'{column}_mo-{lag_month}'))
            for lead_month in range(1, lead):
                new_columns.append(df[column].shift(-lead_month).rename(f'{column}_mo{lead_month}'))
        df_shifted = pd.concat([df] + new_columns, axis=1)
        df_shifted.rename(columns={col: f"{col}_mo0" for col in df.columns}, inplace=True)
        df_shifted = df_shifted.dropna()
        return df_shifted

    def process_files(self, download_dir, mask_file, mask_variables):
        """
        Process GRIB files for a CFS run and insert extracted data into the database.
        """
        # Validate inputs
        if not os.path.isdir(download_dir):
            raise ValueError(f"ERROR: The specified directory does not exist.")
        if not os.path.exists(mask_file):
            raise ValueError("ERROR: mask_file not found.")
        if not isinstance(mask_variables, list):
            raise ValueError("ERROR: mask_variables must be a list of strings.")

        # Load mask and calculate grid area
        mask_ds = nc.Dataset(mask_file)
        mask_lat = mask_ds.variables['latitude'][:]
        mask_lon = mask_ds.variables['longitude'][:]
        area = calculate_grid_cell_areas(mask_lon, mask_lat)
                          
        # Remove .idx files
        for f in os.listdir(download_dir):
            if f.endswith('.idx'):
                os.remove(os.path.join(download_dir, f))

        for filename in sorted(os.listdir(download_dir)):
            file = os.path.join(download_dir, filename)
            parts = filename.split('.')
            forecast = parts[2]
  
            # Extract cfs_run as YYYY-MM-DD HH
            try:
                cfs_dt = datetime.strptime(forecast, '%Y%m%d%H')
                cfs_run = cfs_dt.strftime('%Y-%m-%d %H')
            except ValueError:
                print(f"Skipping file with invalid cfs_run: {filename}")
                continue

            forecast_year = int(forecast[:4])
            forecast_month = int(forecast[4:6])
            _, num_days = calendar.monthrange(forecast_year, forecast_month)

            # ===== Precipitation ===== #
            if filename.startswith('pgbf') and filename.endswith('.grib.grb2'):
                try:
                    pgb_surface = cfgrib.open_dataset(file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'}, decode_timedelta=False)
                    pcp = pgb_surface['tp']
                    pcp_cut = pcp.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max())
                    )
                    pcp_remap = pcp_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')

                    for mask_var in mask_variables:
                        mask = mask_ds.variables[mask_var][:]
                        total_pcp = (np.sum(pcp_remap * mask * area)) * 4 * num_days
                        pcp_mm = total_pcp / np.sum(mask * area)

                        lake_abv, surface_type = mask_var.split('_')
                        lake = {'eri': 'erie', 'ont': 'ontario', 'sup': 'superior', 'mih': 'michigan-huron'}.get(lake_abv)
                        if lake is None:
                            raise ValueError(f"ERROR: The mask variables need to begin with 'eri', 'ont', 'sup', or 'mih'. Check the mask file.")

                        self.db.add(cfs_run, forecast_year, forecast_month, lake, surface_type, 'precipitation', pcp_mm.item())

                except Exception as e:
                    print(f"ERROR processing precipitation data: {e}. Skipping forecast.")
                    continue

            # ===== 2m Temperature ===== #
            elif filename.startswith('flxf') and filename.endswith('.grib.grb2'):
                try:
                    flx_2mabove = cfgrib.open_dataset(file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}, decode_timedelta=False)
                    try:
                        mean2t = flx_2mabove['avg_2t']
                    except KeyError:
                        print("'avg_2t' not found in flux file, trying 'mean2t'.")
                        mean2t = flx_2mabove['mean2t']

                    mean2t_cut = mean2t.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max())
                    )
                    mean2t_remap = mean2t_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')

                    for mask_var in mask_variables:
                        mask = np.ma.masked_where(np.isnan(mask_ds.variables[mask_var][:]), np.ones_like(mask_ds.variables[mask_var][:]))
                        tmp_avg = np.mean(mean2t_remap * mask)

                        lake_abv, surface_type = mask_var.split('_')
                        lake = {'eri': 'erie', 'ont': 'ontario', 'sup': 'superior', 'mih': 'michigan-huron'}.get(lake_abv)
                        if lake is None:
                            raise ValueError(f"ERROR: The mask variables need to begin with 'eri', 'ont', 'sup', or 'mih'. Check the mask file.")

                        self.db.add(cfs_run, forecast_year, forecast_month, lake, surface_type, 'air_temperature', tmp_avg.item())

                except Exception as e:
                    print(f"ERROR processing temperature data: {e}. Skipping forecast.")
                    continue

                # ===== Evaporation ===== #
                try:
                    flx_surface = cfgrib.open_dataset(file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'surface'}, decode_timedelta=False)
                    try:
                        mslhf = flx_surface['avg_slhtf']
                    except KeyError:
                        print("'avg_slhtf' not found in flux file, trying 'mslhf'.")
                        mslhf = flx_surface['mslhf']

                    mslhf_cut = mslhf.sel(
                        latitude=slice(mask_lat.max(), mask_lat.min()),
                        longitude=slice(mask_lon.min(), mask_lon.max())
                    )
                    mslhf_remap = mslhf_cut.interp(latitude=mask_lat, longitude=mask_lon, method='linear')

                    evap = calculate_evaporation(mean2t_remap, mslhf_remap)

                    for mask_var in mask_variables:
                        mask = mask_ds.variables[mask_var][:]
                        total_evap = (np.sum(evap * area * mask)) * num_days * 86400
                        evap_mm = total_evap / np.sum(mask * area)

                        lake_abv, surface_type = mask_var.split('_')
                        lake = {'eri': 'erie', 'ont': 'ontario', 'sup': 'superior', 'mih': 'michigan-huron'}.get(lake_abv)
                        if lake is None:
                            raise ValueError(f"ERROR: The mask variables need to begin with 'eri', 'ont', 'sup', or 'mih'. Check the mask file.")

                        self.db.add(cfs_run, forecast_year, forecast_month, lake, surface_type, 'evaporation', evap_mm.item())

                except Exception as e:
                    print(f"ERROR processing evaporation data: {e}. Skipping forecast.")
                    continue
            else:
                print(f"Skipping unrecognized file: {filename}")
                continue

    @staticmethod
    def structure_input(data):
        """
        Transforms a long-format CFS forecast DataFrame into a wide-format one,
        with one column per lake/surface/component/forecast_month combination,
        and dummy variables for initialization month.

        Parameters:
            data (pd.DataFrame): Input DataFrame with columns:
                ['date', 'year', 'month', 'lake', 'surface_type', 'component', 'value [mm]']

        Returns:
            pd.DataFrame: Transformed wide-format DataFrame with dummy-encoded init months.
        """
        data['cfs_run'] = pd.to_datetime(data['cfs_run'], format='%Y-%m-%d %H')

        # Create a datetime column from the 'year' and 'month' columns (set day to 1)
        data['forecast_date'] = pd.to_datetime(dict(year=data['year'], month=data['month'], day=1))

        # Calculate the lead time in months
        data['forecast_month'] = (data['forecast_date'].dt.year - data['cfs_run'].dt.year) * 12 + \
                            (data['forecast_date'].dt.month - data['cfs_run'].dt.month)

        # Drop the intermediate column if you don't need it
        data.drop(columns='forecast_date', inplace=True)

        # Create column names
        data['column_name'] = (
            data['lake'] + '_' +
            data['surface_type'] + '_' +
            data['component'] + '_mo' +
            data['forecast_month'].astype(str)
        )

        # Pivot the DataFrame to wide format
        df_wide = data.pivot(index='cfs_run', columns='column_name', values='value [mm]')

        # Remove any columns ending in '_month10'
        df_wide = df_wide.loc[:, ~df_wide.columns.str.endswith('_mo10')]
        df_wide.dropna(inplace=True)
        # Remove column level name
        df_wide.columns.name = None

        # Extract the month
        df_wide['init_month'] = df_wide.index.month
        df_wide['init_month'] = pd.Categorical(df_wide['init_month'], categories=range(1, 13))

        df = pd.get_dummies(df_wide, columns=['init_month'], prefix='month')

        feature_column_order = (
            [f'month_{i}' for i in range(1, 13)] +
            [
                f'{lake}_{surface_type}_{comp}_mo{m}'
                for lake in ['superior', 'michigan-huron', 'erie', 'ontario']
                for surface_type in ['lake', 'land']
                for comp in ['precipitation', 'evaporation', 'air_temperature']
                for m in range(10) #from 0 to 9, representing the 10 months of lead time
            ]
        )

        df_final = df[feature_column_order]

        return df_final

class CNBSForecaster:
    def __init__(self, model_dir, scaler_dir):
        """
        Initialize the ForecastModel class.

        Parameters
        ----------
        model_dir : str
            Path to directory containing trained model files (e.g., *.joblib).
        scaler_dir : str
            Path to directory containing x_scaler.joblib and y_scaler.joblib.
        """
        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        self.x_scaler = joblib.load(os.path.join(scaler_dir, "x_scaler.joblib"))
        self.y_scaler = joblib.load(os.path.join(scaler_dir, "y_scaler.joblib"))
        self.models = self._load_models()

    def _load_models(self):
        """
        Load all models from the model directory.

        Returns
        -------
        dict
            Dictionary of model name → model object
        """
        models = {}
        for file in os.listdir(self.model_dir):
            if file.endswith("_trained_model.joblib"):
                model_name = file.split("_")[0]  # Assumes filename starts with model name
                model_path = os.path.join(self.model_dir, file)
                models[model_name] = joblib.load(model_path)
        return models

    def predict(self, X, model_name):
        """
        Predict CNBS values using a specified model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features to predict on.
        model_name : str
            Name of the model to use (e.g., 'RF', 'GP').

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in {self.model_dir}")

        model = self.models[model_name]
        X_scaled = self.x_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        # Define column names
        column_names = [
            f"{lake}_{comp}_mo{m}"
            for lake in ['superior', 'michigan-huron', 'erie', 'ontario']
            for comp in ['precipitation', 'evaporation', 'runoff', 'nbs']
            for m in range(12)
        ]

        df = pd.DataFrame(y, columns=column_names, index=X.index)
        df["model"] = model_name
        return df

class ForecastTransformer:
    """
    Transforms CFS forecast data from wide to long format,
    and filters based on forecast month.
    """

    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df.copy()

    def melt(self):
        """
        Transforms the forecast DataFrame into long format and then pivots it
        by lake, component, and forecast month for easier analysis or storage.
        """
        df = self.df.copy()

        # Step 1: Define ID vars and melt the rest
        id_vars = ['cfs_run', 'model']
        value_vars = [col for col in df.columns if col not in id_vars]

        df_melt = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='lake_component_month',
            value_name='value'
        )

        # Step 2: Split lake_component_month into lake, component, and month
        df_melt[['lake', 'component', 'month']] = df_melt['lake_component_month'].str.rsplit('_', n=2, expand=True)

        # Step 3: Clean up the month column
        df_melt['month'] = df_melt['month'].str.replace('mo', 'month_', regex=False)

        # Step 4: Pivot to wide format with months as columns
        df_final = df_melt.pivot_table(
            index=id_vars + ['lake', 'component'],
            columns='month',
            values='value'
        ).reset_index()
        df_final.columns.name = None  # Remove the columns name

        # Step 5: Sort month columns numerically
        month_cols = sorted(
            [col for col in df_final.columns if col.startswith('month_')],
            key=lambda x: int(x.split('_')[1])
        )

        # Final column ordering
        df_final = df_final[id_vars + ['lake', 'component'] + month_cols]

        # Return the DataFrame
        return df_final

    def pivot(self):
        """
        Reshape forecast DataFrame so each lake & variable type
        gets its own column with forecast_month calculated.
        """
        df = self.df.copy()

        # Step 1: Melt into long format
        df_melted = df.melt(
            id_vars=['cfs_run', 'model'],
            var_name='variable',
            value_name='value'
        )

        # Step 2: Split into lake, variable, and month_offset
        split_cols = df_melted['variable'].str.rsplit('_', n=2, expand=True)
        df_melted['lake'] = split_cols[0]
        df_melted['variable'] = split_cols[1]
        df_melted['month_offset'] = split_cols[2].str.replace('mo', '', regex=False).astype(int)

        # Step 3: Compute forecast_month
        df_melted['forecast_month'] = df_melted.apply(
            lambda row: (pd.to_datetime(row['cfs_run']) + pd.DateOffset(months=row['month_offset'])).strftime('%Y-%m'),
            axis=1
        )
        df_melted = df_melted.drop(columns=['month_offset'])

        # Step 4: Pivot so each variable type is its own column
        df_tidy = df_melted.pivot_table(
            index=['cfs_run', 'model', 'forecast_month', 'lake'],
            columns='variable',
            values='value'
        ).reset_index()
        df_tidy.columns.name = None

        # Step 5: Reorder columns if expected vars exist
        expected_vars = ["precipitation", "evaporation", "runoff", "nbs"]
        cols = ["cfs_run", "forecast_month", "model", "lake"] + [v for v in expected_vars if v in df_tidy.columns]
        df_reordered = df_tidy[cols]

        df_final = df_reordered

        return df_final


    def filter(self, first_forecast_month):
        """
        Filters rows where forecast_month is greater than or equal to a minimum (YYYY-MM).
        """
        df = self.df.copy()
        df['forecast_month'] = pd.to_datetime(df['forecast_month'], format="%Y-%m")
        min_date = pd.to_datetime(first_forecast_month, format="%Y-%m")
        df = df[df['forecast_month'] >= min_date]
        df['forecast_month'] = df['forecast_month'].dt.to_period('M').astype(str)
        return df
