import os
import pandas as pd
import cfgrib
import sqlite3
import numpy as np
import calendar
from datetime import datetime
import joblib
import netCDF4 as nc
import json
import uuid
import re
from typing import Optional, Sequence, Dict
from properscoring import crps_ensemble
from sklearn.metrics import mean_squared_error, r2_score

from src.database_utils import CFSDatabase
from src.hydro_utils import calculate_evaporation_rate, calculate_grid_cell_areas

class CFSProcessor:
    def __init__(self, database, table):
        """
        Initialize the processor with database path and table name.
        """
        self.database = database
        self.table = table
        self.db = CFSDatabase(database, table)

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
            cfs_run = parts[2]

            forecast_year = int(parts[3][:4])
            forecast_month = int(parts[3][4:6])
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

                    evap = calculate_evaporation_rate(mean2t_remap, mslhf_remap)

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

class SeasonalCycleProcessor:
    """
    SeasonalCycleProcessor

    Computes and applies a monthly climatology to a pandas DataFrame
    with a DatetimeIndex at monthly frequency.

    This class supports:
        - fit(): compute monthly climatology from a baseline period
        - transform(): convert raw values -> anomalies
        - inverse_transform(): convert anomalies -> raw values
        - save()/load(): persist climatology artifact to disk

    Assumptions
    -----------
    - Input DataFrames must have a pandas.DatetimeIndex.
    - Index must represent monthly timestamps.
    - Monthly climatology is computed as the mean for each calendar month (1–12).
    """

    def __init__(self):
        self.climatology: Optional[pd.DataFrame] = None
        self.metadata: Dict = {
            "id": str(uuid.uuid4())
        }

    # ---------------------------------------------------------------------
    # FIT
    # ---------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        var_list: Optional[Sequence[str]] = None,
        baseline_time: Optional[slice] = None,
        baseline_definition: Optional[Dict] = None,
    ) -> "SeasonalCycleProcessor":
        """
        Compute monthly climatology from a baseline period.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame with a DatetimeIndex.
            Rows represent monthly observations.
            Columns represent variables.

        var_list : sequence of str, optional
            Subset of columns to compute climatology for.
            If None, all numeric columns are used.

        baseline_time : slice, optional
            Time slice applied before computing climatology.
            Example:
                slice("1981-01-01", "2008-12-01")

            If None, full DataFrame is used.

        baseline_definition : dict, optional
            Metadata describing baseline choice (for reproducibility).
            Example:
                {"train_start": "1981-01-01", "train_end": "2008-12-01"}

        Returns
        -------
        self : SeasonalCycleProcessor
            Fitted processor with climatology stored internally.
        """

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        if var_list is None:
            var_list = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        missing = [c for c in var_list if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        fit_df = df
        if baseline_time is not None:
            fit_df = df.loc[baseline_time]

        if fit_df.empty:
            raise ValueError("Baseline selection resulted in empty DataFrame.")

        # Compute monthly mean climatology
        months = fit_df.index.month
        climatology = fit_df[var_list].groupby(months).mean()
        climatology.index.name = "month"

        # Ensure months 1–12 are present
        climatology = climatology.reindex(range(1, 13))
        print(climatology)

        self.climatology = climatology

        # Store metadata
        self.metadata.update({
            "var_list": list(var_list),
            "baseline_definition": baseline_definition,
            "calculation_date": str(pd.Timestamp.now()),
            "reduction": "monthly_mean",
        })

        return self

    # ---------------------------------------------------------------------
    # TRANSFORM
    # ---------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw values to anomalies.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with DatetimeIndex.
            Must contain columns used during fit().

        Returns
        -------
        anomalies : pandas.DataFrame
            DataFrame with same shape as input,
            where climatological monthly means have been subtracted.
        """

        if self.climatology is None:
            raise ValueError("Processor must be fitted before calling transform().")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        var_list = self.metadata["var_list"]
        months = df.index.month

        out = df.copy()
        out[var_list] = (
            out[var_list].to_numpy()
            - self.climatology.loc[months, var_list].to_numpy()
        )

        return out

    # ---------------------------------------------------------------------
    # INVERSE TRANSFORM
    # ---------------------------------------------------------------------

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert anomalies back to raw values.

        Parameters
        ----------
        df : pandas.DataFrame
            Anomaly DataFrame with DatetimeIndex.

        Returns
        -------
        raw : pandas.DataFrame
            DataFrame with climatology added back.
        """

        if self.climatology is None:
            raise ValueError("Processor must be fitted before calling inverse_transform().")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex.")

        var_list = self.metadata["var_list"]
        months = df.index.month

        out = df.copy()
        out[var_list] = (
            out[var_list].to_numpy()
            + self.climatology.loc[months, var_list].to_numpy()
        )

        return out

    # ---------------------------------------------------------------------
    # SAVE / LOAD
    # ---------------------------------------------------------------------

    def save(self, base_dir: str = "seasonal_cycles") -> Dict[str, str]:
        """
        Save climatology artifact to disk.

        Parameters
        ----------
        base_dir : str
            Directory where artifact files will be written.

        Returns
        -------
        dict
            Dictionary containing:
                {
                    "climatology_path": <path>,
                    "metadata_path": <path>
                }
        """

        if self.climatology is None:
            raise ValueError("Nothing to save. Call fit() first.")

        os.makedirs(base_dir, exist_ok=True)

        clim_path = os.path.join(base_dir, "climatology.csv")
        meta_path = os.path.join(base_dir, "metadata.json")

        self.climatology.to_csv(clim_path)

        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return {
            "climatology_path": clim_path,
            "metadata_path": meta_path,
        }

    @classmethod
    def load(cls, climatology_path: str, metadata_path: str) -> "SeasonalCycleProcessor":
        """
        Load a previously saved climatology artifact.

        Parameters
        ----------
        climatology_path : str
            Path to saved climatology CSV.

        metadata_path : str
            Path to saved metadata JSON.

        Returns
        -------
        SeasonalCycleProcessor
            Processor with climatology loaded.
        """

        instance = cls()

        instance.climatology = pd.read_csv(
            climatology_path,
            index_col=0
        )
        instance.climatology.index = instance.climatology.index.astype(int)
        instance.climatology.index.name = "month"

        with open(metadata_path, "r") as f:
            instance.metadata = json.load(f)

        return instance

    @staticmethod
    def build_climatology_variable_name_long(
        df: pd.DataFrame,
        *,
        lake_col: str = "lake",
        surface_col: str = "surface",
        component_col: str = "component",
    ) -> pd.Series:
        """
        Build climatology variable names for long-format operational CFS forecast data.

        Expected output names match training climatology columns, e.g.:
            superior_lake_precipitation
            michigan-huron_land_air_temperature
        """
        required = [lake_col, surface_col, component_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return (
            df[lake_col].astype(str)
            + "_"
            + df[surface_col].astype(str)
            + "_"
            + df[component_col].astype(str)
        )

    @staticmethod
    def subtract_climatology_long(
        df: pd.DataFrame,
        scp: "SeasonalCycleProcessor",
        *,
        value_col: str = "value",
        month_col: str = "month",
        lake_col: str = "lake",
        surface_col: str = "surface_type",
        component_col: str = "component",
        output_col: str = "value_anom",
        variable_col: str = "climatology_variable",
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Convert absolute long-format operational CFS forecast values to anomalies.

        This is intended for data from cfs_forecast_data.db, where each row contains:
            cfs_run, year, month, lake, surface_type, component, value

        Unlike lead-wide training data, the operational table already contains the
        verifying forecast month in `month`, so no init-date + lead calculation is needed.

        Operation:
            value_anom = value - climatology[month, variable]

        where:
            variable = f"{lake}_{surface_type}_{component}"
        """
        if scp.climatology is None:
            raise ValueError("SeasonalCycleProcessor must be fitted or loaded before use.")

        required = [value_col, month_col, lake_col, surface_col, component_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        clim = scp.climatology
        out = df.copy()

        out[month_col] = out[month_col].astype(int)
        out[variable_col] = SeasonalCycleProcessor.build_climatology_variable_name_long(
            out,
            lake_col=lake_col,
            surface_col=surface_col,
            component_col=component_col,
        )

        bad_months = sorted(set(out[month_col]) - set(range(1, 13)))
        if bad_months:
            raise ValueError(f"{month_col} must contain values 1–12. Got: {bad_months}")

        missing_vars = sorted(set(out[variable_col]) - set(clim.columns))
        if missing_vars and strict:
            raise KeyError(
                "Some operational forecast variables are missing from the climatology: "
                f"{missing_vars}"
            )

        if missing_vars:
            out[output_col] = np.nan
            ok = out[variable_col].isin(clim.columns)
        else:
            ok = pd.Series(True, index=out.index)

        clim_values = pd.Series(np.nan, index=out.index, dtype=float)

        for var in out.loc[ok, variable_col].unique():
            rows = ok & (out[variable_col] == var)
            months = out.loc[rows, month_col].to_numpy()
            clim_values.loc[rows] = clim.loc[months, var].to_numpy()

        out[output_col] = out[value_col].to_numpy() - clim_values.to_numpy()

        return out
    # -----------------------------------------------------------------------------
    # Lead-aware utilities (targets shifted to *_mo{k})
    # -----------------------------------------------------------------------------

    _MO_RE = re.compile(r"_mo(\d+)$")

    @staticmethod
    def subtract_climatology_leadwide(
        df_abs_leadwide: pd.DataFrame,
        scp_X,
        strict: bool = False,
    ) -> pd.DataFrame:

        """
        Convert lead-wide absolute forecast values into climatological anomalies
        by subtracting the monthly climatology corresponding to each forecast
        verifying month.

        This function is designed for datasets containing multiple forecast lead
        months stored as separate columns using the naming convention:

            <variable>_mo{k}

        where:
            - <variable> is the base predictor variable name
            - k is the forecast lead month

        Examples
        --------
        superior_precipitation_mo0
        superior_precipitation_mo1
        erie_runoff_mo3

        For each column:

        1. The forecast lead month is extracted using the `_mo{k}` suffix.
        2. The base variable name is identified by removing the suffix.
        3. The verifying month is computed by shifting the initialization date
        forward by the forecast lead.
        4. The corresponding monthly climatology is subtracted from the forecast
        value.

        The verifying month is calculated as:

            verifying_month =
                initialization_date + forecast_lead_months

        This ensures that climatology subtraction is aligned with the actual
        target month being forecasted rather than the initialization month.

        If a column does not contain a `_mo{k}` suffix, it is treated as lead 0.

        Parameters
        ----------
        df_abs_leadwide : pd.DataFrame
            DataFrame containing absolute forecast values indexed by forecast
            initialization date. The index must be a DatetimeIndex.

        scp_X : SeasonalCycleProcessor
            A fitted SeasonalCycleProcessor object containing a `climatology`
            attribute. The climatology DataFrame must be indexed by month
            (1-12) and contain columns matching the base variable names.

        strict : bool, optional
            If True, raise a KeyError when a climatology variable is not found.
            If False, missing variables are skipped with a warning message.
            Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same shape, index, and columns as the input,
            but with values transformed from absolute space into anomaly space
            by subtracting the monthly climatology.

        Raises
        ------
        ValueError
            If `df_abs_leadwide` does not have a DatetimeIndex.

        AttributeError
            If `scp_X` does not contain a `climatology` attribute.

        KeyError
            If `strict=True` and a base variable is not found in the
            climatology columns.
        """
        if not isinstance(df_abs_leadwide.index, pd.DatetimeIndex):
            raise ValueError("df_abs_leadwide must have a DatetimeIndex.")

        if not hasattr(scp_X, "climatology"):
            raise AttributeError("scp_X must have a `climatology` attribute.")

        clim = scp_X.climatology.copy()
        clim.columns = clim.columns.astype(str).str.strip()

        out = df_abs_leadwide.copy()
        out.columns = out.columns.astype(str).str.strip()

        idx = out.index

        _MO_RE = re.compile(r"_mo(\d+)$")

        for col in out.columns:
            m = _MO_RE.search(col)

            if m is not None:
                lead = int(m.group(1))
                base = _MO_RE.sub("", col).strip()
            else:
                lead = 0
                base = col.strip()

            if base not in clim.columns:
                if strict:
                    raise KeyError(
                        f"Base variable '{base}' not found in climatology columns."
                    )
                else:
                    print(f"Skipping '{col}' because '{base}' is not in climatology.")
                    continue

            verifying_month = (idx + pd.DateOffset(months=lead)).month

            out[col] = (
                out[col].to_numpy()
                - clim.loc[verifying_month, base].to_numpy()
            )

        return out

    @staticmethod
    def add_climatology_back_leadwide(
        df_anom_leadwide: pd.DataFrame,
        scp_y: "SeasonalCycleProcessor",
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Convert lead-wide anomaly targets to lead-wide absolute targets by adding a monthly climatology.

        Parameters
        ----------
        df_anom_leadwide : pd.DataFrame
            Anomaly DataFrame indexed by init date (must be a DatetimeIndex).
        scp_y : SeasonalCycleProcessor
            A fitted seasonal-cycle processor.
        strict : bool
            If True, raise error if column does not match `_mo{k}`.

        Returns
        -------
        pd.DataFrame
            Same shape/columns/index as `df_anom_leadwide`, but in absolute units.
        """
        if not isinstance(df_anom_leadwide.index, pd.DatetimeIndex):
            raise ValueError("df_anom_leadwide must have a DatetimeIndex (init dates).")

        if not hasattr(scp_y, "climatology"):
            raise AttributeError("scp_y must have a `climatology` attribute (fit the processor first).")

        clim = scp_y.climatology
        out = df_anom_leadwide.copy()
        idx = out.index

        for col in out.columns:
            m = SeasonalCycleProcessor._MO_RE.search(col)
            if m is None:
                if strict:
                    raise ValueError(f"Column '{col}' does not end with _mo{{k}}.")
                continue
            lead = int(m.group(1))
            base = SeasonalCycleProcessor._MO_RE.sub("", col)
            if base not in clim.columns:
                raise KeyError(f"Base variable '{base}' not found in climatology columns.")
            verifying_month = (idx + pd.DateOffset(months=lead)).month
            out[col] = out[col].to_numpy() + clim.loc[verifying_month, base].to_numpy()
        return out
    
    @staticmethod
    def load_clim(directory):
        """
        Load SeasonalCycleProcessor objects for inputs and targets from a directory.

        Parameters
        ----------
        directory : str
            Base directory containing `inputs` and `targets` subfolders with
            climatology CSVs and metadata JSONs.

        Returns
        -------
        scp_X : SeasonalCycleProcessor
            Processor for input features.
        scp_y : SeasonalCycleProcessor
            Processor for target outputs.
        """
        import os

        # Ensure directory ends with separator
        directory = os.path.join(directory, '')

        scp_X = SeasonalCycleProcessor.load(
            climatology_path=os.path.join(directory, "inputs", "climatology.csv"),
            metadata_path=os.path.join(directory, "inputs", "metadata.json")
        )

        scp_y = SeasonalCycleProcessor.load(
            climatology_path=os.path.join(directory, "targets", "climatology.csv"),
            metadata_path=os.path.join(directory, "targets", "metadata.json")
        )

        return scp_X, scp_y

class CFSTransformer:
    def __init__(self, df):
        """
        Initialize the transformer with a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = df.copy()


    def filter(self, today=None, months_back=9):
        """
        Filter rows to keep only CFS runs going back a specified number
        of months from the current operational forecast month.

        Rules
        -----
        - If today's day is before the 26th, use the current month.
        - If today's day is on or after the 26th, use the following month.
        - Then go back `months_back` months.
        - Keep rows where cfs_run is greater than or equal to that start date.

        Parameters
        ----------
        today : datetime, optional
            Reference date. Defaults to current date if None.

        months_back : int, default=9
            Number of months to go back from the first forecast month.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """

        df = self.df.copy()

        if today is None:
            today = datetime.today()

        # Determine operational forecast month
        if today.day < 26:
            forecast_month = datetime(today.year, today.month, 1)
        else:
            forecast_month = (
                pd.Timestamp(today.year, today.month, 1)
                + pd.DateOffset(months=1)
            )

        # Go back desired number of months
        start_date = (
            pd.Timestamp(forecast_month)
            - pd.DateOffset(months=months_back)
        )

        # Convert cfs_run to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["cfs_run"]):
            df["cfs_run"] = pd.to_datetime(
                df["cfs_run"].astype(str),
                format="%Y%m%d%H"
            )

        # Filter
        df_filtered = df[df["cfs_run"] >= start_date].copy()

        print(f"Beginning from month {pd.Timestamp(forecast_month).strftime('%m-%Y')}")

        return df_filtered

    def shift_variables(self, lag=0, lead=0):
        """
        Create the variables columns to include lags (last month values) and lead variables
        
        Parameters:
        - df (pd.DataFrame): The DataFrame containing the time series data.
        - lag (int): The number of months you want to include lagged variables. Default = 0
        - lead (int): The number of months for the advance variables. Default = 0
        
        Returns:
        - pd.DataFrame: The DataFrame with added variable columns for lags and leading.
        """
        df = self.df.copy()  # To avoid modifying the original DataFrame

        new_columns = []  # List to store the new lag and lead columns

        # Generate target columns for the lag and lead months
        for column in df.columns:
            
            for lag_month in range(1, lag):
                new_columns.append(df[column].shift(lag_month).rename(f'{column}_mo-{lag_month}'))
            for lead_month in range(1, lead):
                new_columns.append(df[column].shift(-lead_month).rename(f'{column}_mo{lead_month}'))

        # Concatenate the new columns with the original DataFrame
        df_shifted = pd.concat([df] + new_columns, axis=1)

        # Rename original columns to have _mo0 suffix
        df_shifted.rename(columns={col: f"{col}_mo0" for col in df.columns}, inplace=True)

        # Drop rows with any NaN values generated by shifting for the target
        df_shifted = df_shifted.dropna()

        return df_shifted

    def structure_input(self):
        """
        Transforms a long-format CFS forecast DataFrame into a wide-format one.

        Parameters:
            mode (str): "absolute" or "anomaly"
            scp (SeasonalCycleProcessor): required if mode="anomaly"

        Returns:
            pd.DataFrame: A wide-format DataFrame where each column corresponds to a specific lake, 
            variable type, and forecast month.
        """
        data = self.df.copy()

        # In the past I have used "value [mm]" and tried to switch it to "value"
        value_col = "value [mm]" if "value [mm]" in data.columns else "value"

        data['cfs_run'] = pd.to_datetime(data['cfs_run'], format='%Y%m%d%H')

        data['forecast_date'] = pd.to_datetime(
            dict(year=data['year'], month=data['month'], day=1)
        )

        data['forecast_month'] = (
            (data['forecast_date'].dt.year - data['cfs_run'].dt.year) * 12 +
            (data['forecast_date'].dt.month - data['cfs_run'].dt.month)
        )

        data.drop(columns='forecast_date', inplace=True)

        # Column naming
        data['column_name'] = (
            data['lake'] + '_' +
            data['surface_type'] + '_' +
            data['component'] + '_mo' +
            data['forecast_month'].astype(str)
        )

        # Use value_col instead of hardcoded column
        df_wide = data.pivot(
            index='cfs_run',
            columns='column_name',
            values=value_col
        )

        df_wide = df_wide.loc[:, ~df_wide.columns.str.endswith('_mo10')]
        df_wide.dropna(inplace=True)
        df_wide.columns.name = None

        feature_column_order = (
            [
                f'{lake}_{surface_type}_{comp}_mo{m}'
                for lake in ['superior', 'michigan-huron', 'erie', 'ontario']
                for surface_type in ['lake', 'land']
                for comp in ['precipitation', 'evaporation', 'air_temperature']
                for m in range(10)
            ]
        )

        df_final = df_wide[feature_column_order]

        return df_final

    def add_time_features(
        self,
        add_month_cycle=True,
        add_time_trend=True,
        normalize_time=True,
        overwrite=True,
        copy=True
    ):
        """
        Add cyclical month features and a continuous time trend.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe indexed by a pandas DatetimeIndex.

        add_month_cycle : bool, default=True
            If True, add cyclical month features:
                - month_sin
                - month_cos

            These preserve the cyclical nature of calendar months
            (e.g., December is close to January).

        add_time_trend : bool, default=True
            If True, add a continuous decimal-year time trend feature.

        normalize_time : bool, default=True
            If True, standardize the time feature to:
                mean = 0
                standard deviation = 1

            This improves numerical stability for many machine
            learning models.

        overwrite : bool, default=True
            If True, remove any pre-existing:
                - month_sin
                - month_cos
                - time

            before adding new versions.

        copy : bool, default=True
            If True, operate on a copy of the dataframe.

        Returns
        -------
        pandas.DataFrame
            Dataframe with added temporal predictor columns.

        Notes
        -----
        The cyclical month encoding is defined as:

            month_sin = sin(2π * month / 12)
            month_cos = cos(2π * month / 12)

        which avoids discontinuities between December and January.

        The continuous time trend is represented as:

            year + (month - 1) / 12

        and optionally normalized.
        """

        # ---------------------------------------------------------
        # Optionally work on a copy to avoid modifying original data
        # ---------------------------------------------------------
        out = self.df.copy()

        # ---------------------------------------------------------
        # Ensure dataframe uses a DatetimeIndex
        # ---------------------------------------------------------
        if not isinstance(out.index, pd.DatetimeIndex):
            raise TypeError(
                "DataFrame index must be a pandas DatetimeIndex."
            )

        # ---------------------------------------------------------
        # Remove existing temporal feature columns if requested
        # ---------------------------------------------------------
        temporal_cols = ["month_sin", "month_cos", "time"]

        if overwrite:
            out = out.drop(
                columns=[c for c in temporal_cols if c in out.columns],
                errors="ignore"
            )

        # =========================================================
        # CYCLICAL MONTH FEATURES
        # =========================================================
        if add_month_cycle:

            # Convert months to numpy array for vectorized operations
            month = out.index.month.to_numpy()

            # Encode month as cyclical sine/cosine coordinates
            out["month_sin"] = np.sin(
                2 * np.pi * month / 12
            )

            out["month_cos"] = np.cos(
                2 * np.pi * month / 12
            )

        # =========================================================
        # CONTINUOUS TIME TREND
        # =========================================================
        if add_time_trend:

            # Decimal year representation
            #
            # Example:
            #   Jan 2015 -> 2015.00
            #   Feb 2015 -> 2015.08
            #   Jul 2015 -> 2015.50
            #
            t = (
                out.index.year.to_numpy()
                + (out.index.month.to_numpy() - 1) / 12.0
            )

            # Shift to start at zero for numerical stability
            time = t - t.min()

            # -----------------------------------------------------
            # Optional standardization
            # -----------------------------------------------------
            if normalize_time:

                time_std = time.std()

                # Avoid divide-by-zero for constant arrays
                if time_std == 0:
                    time = np.zeros_like(time)

                else:
                    time = (
                        (time - time.mean())
                        / time_std
                    )

            out["time"] = time

        return out
        
class CNBSForecaster:
    def __init__(self, model_dir, scaler_dir, mode="actual"):
        """
        Initialize the ForecastModel class.

        Parameters
        ----------
        model_dir : str
        scaler_dir : str
        mode : str
            "actual" or "anomaly"
        """
        if mode not in ["actual", "anomaly"]:
            raise ValueError("mode must be 'actual' or 'anomaly'")

        self.model_dir = model_dir
        self.scaler_dir = scaler_dir
        self.mode = mode

        # Set suffixes
        suffix = "_anom" if mode == "anomaly" else ""

        # Load scalers
        self.x_scaler = joblib.load(
            os.path.join(scaler_dir, f"x_scaler{suffix}.joblib")
        )
        self.y_scaler = joblib.load(
            os.path.join(scaler_dir, f"y_scaler{suffix}.joblib")
        )

        # Load models
        self.models = self._load_models(suffix)

    def _load_models(self, suffix):
        models = {}
        for file in os.listdir(self.model_dir):
            if file.endswith(f"_trained_model{suffix}.joblib"):
                model_name = file.split("_")[0]
                model_path = os.path.join(self.model_dir, file)
                models[model_name] = joblib.load(model_path)
        return models

    def predict(self, X, model_name, scp_y=None):
        """
        Predict CNBS values.

        Parameters
        ----------
        X : pd.DataFrame
        model_name : str
        mode : str
            "actual" or "anomaly"
        scp_y : object, optional
            Required if mode='anomaly' to convert anomalies back to absolute

        Returns
        -------
        pd.DataFrame
        """

        mode = self.mode

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        if mode == "anomaly" and scp_y is None:
            raise ValueError("scp_y must be provided when mode='anomaly'")

        model = self.models[model_name]

        # --- Scale and predict ---
        X_scaled = self.x_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        y = self.y_scaler.inverse_transform(y_scaled)

        # --- Generate column names ---
        column_names = [
            f"{lake}_{comp}_mo{m}"
            for lake in ['superior', 'michigan-huron', 'erie', 'ontario']
            for comp in ['precipitation', 'evaporation', 'runoff', 'nbs']
            for m in range(12)
        ]

        # --- Build DataFrame ---
        df_pred = pd.DataFrame(y, columns=column_names, index=X.index)

        # --- Convert anomalies to absolute if needed ---
        if mode == "anomaly":
            df_pred = SeasonalCycleProcessor.add_climatology_back_leadwide(df_pred, scp_y)

        # --- Add model column efficiently ---
        df_model = pd.DataFrame(
            {"model": [model_name] * len(df_pred)},
            index=df_pred.index
        )

        # Concatenate all at once to avoid fragmentation
        df_final = pd.concat([df_pred, df_model], axis=1)

        return df_final

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
        df["cfs_run"] = pd.to_datetime(df["cfs_run"], format ="%Y%m%d%H")

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
        df_final = df_tidy[cols]

        # Convert cfs_run back to YYYYMMDDHH format
        df_final["cfs_run"] = pd.to_datetime(df_final["cfs_run"]).dt.strftime("%Y%m%d%H")

        return df_final

    def filter(self, first_forecast_month):
        """
        Filters rows to keep the first forecast month >= first_forecast_month
        and all subsequent months.
        
        Works whether dataframe has 'forecast_month' (YYYY-MM) or 'year' + 'month'.
        """
        df = self.df.copy()

        # Convert threshold to a Period
        min_period = pd.Period(first_forecast_month, freq="M")

        if "forecast_month" in df.columns:
            # Convert to Period
            df["forecast_month"] = pd.to_datetime(df["forecast_month"], format="%Y-%m").dt.to_period("M")

            # Find the first forecast month >= threshold
            first_month_in_data = df["forecast_month"].min()
            first_month_to_keep = max(first_month_in_data, min_period)

            # Keep rows >= that first month
            df = df[df["forecast_month"] >= first_month_to_keep]

            # Convert back to string YYYY-MM if needed
            df["forecast_month"] = df["forecast_month"].astype(str)
            return df

        elif "year" in df.columns and "month" in df.columns:
            # Build Period
            df["forecast_period"] = df.apply(lambda r: pd.Period(f"{r.year}-{r.month:02d}", freq="M"), axis=1)

            first_month_in_data = df["forecast_period"].min()
            first_month_to_keep = max(first_month_in_data, min_period)

            df = df[df["forecast_period"] >= first_month_to_keep]

            # Optionally drop helper column
            df = df.drop(columns=["forecast_period"])
            return df

        else:
            raise ValueError("Dataframe must contain either 'forecast_month' or ('year','month') columns.")

@staticmethod
def align_prob_with_start_date(merged_df, start_date):
    """
    Aligns the months in a lake probability DataFrame to a specified start date,
    creating a continuous datetime index beginning at that month.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame containing at least the columns:
        - 'month' (str): three-letter month abbreviation ("Jan"–"Dec")
        - optionally 'month_num' and 'year' (they will be rebuilt if missing)
    start_date : str
        Start date in "MM-YYYY" format (e.g., "11-2025").
        The resulting index will begin at this month.

    Returns
    -------
    pd.DataFrame
        Copy of merged_df with an added 'date' column (and datetime index)
        spanning one full 12-month cycle starting from `start_date`.
        The DataFrame is sorted chronologically by this new index.

    Example
    -------
    >>> prob_aligned = align_prob_with_start_date(prob, "11-2025")
    >>> prob_aligned.query("lake == 'erie'").head()
               month   lake  prob_exceedance     value
    date
    2025-11-01   Nov   erie              0.5   85.4
    2025-12-01   Dec   erie              0.5  102.2
    2026-01-01   Jan   erie              0.5  125.8
    """
    df = merged_df.copy()

    # Define month mapping
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_to_num = {m: i + 1 for i, m in enumerate(month_order)}

    # Add month number if missing
    if "month_num" not in df.columns:
        df["month_num"] = df["month"].map(month_to_num)

    # Parse the start date (e.g., 11-2025 → month=11, year=2025)
    start_month, start_year = map(int, start_date.split("-"))

    # Compute year assignment for each month
    # Months >= start_month → same year, months < start_month → next year
    df["year"] = df["month_num"].apply(
        lambda m: start_year if m >= start_month else start_year + 1
    )

    # Build datetime column
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month_num"], day=1)
    )

    # Sort chronologically and set index
    df = df.sort_values(["lake", "date"]).set_index("date")

    # Drop helper columns
    df = df.drop(columns=["month_num", "year"], errors="ignore").reset_index()
    
    return df

@staticmethod
def create_accumulation_dataframe(
    df,
    accumulation_periods=(3, 6),
    exclude_columns=None,
):
    """
    Create a wide-format accumulation dataframe with one row per
    cfs_run and model.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format forecast dataframe containing:
            - cfs_run
            - model
            - forecast_month

    accumulation_periods : tuple, default=(3, 6)
        Number of forecast months to accumulate.

    exclude_columns : list or None
        Columns to exclude from accumulation calculations.

    Returns
    -------
    pandas.DataFrame
        Wide-format accumulation dataframe.
    """

    df = df.copy()

    df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    df["cfs_run"] = pd.to_datetime(df["cfs_run"])

    # Sort by lead time
    df = df.sort_values(
        ["cfs_run", "model", "forecast_month"]
    )

    # Columns not to accumulate
    default_exclude = {
        "cfs_run",
        "forecast_month",
        "model",
        "date",
    }

    if exclude_columns is not None:
        default_exclude.update(exclude_columns)

    value_columns = [
        c for c in df.columns
        if c not in default_exclude
    ]

    output_rows = []
    grouped = df.groupby(["cfs_run", "model"])

    for (cfs_run, model), group in grouped:

        row = {
            "cfs_run": cfs_run,
            "model": model,
        }

        group = group.sort_values("forecast_month")

        for acc in accumulation_periods:
            subset = group.iloc[:acc]
            for col in value_columns:
                row[f"{col}_acc{acc}"] = subset[col].sum()

        output_rows.append(row)

    return pd.DataFrame(output_rows)

def compute_skill(
    df,
    lakes,
    components,
    model,
    accumulations=None,   # None = monthly, e.g. (3, 6) = accumulated
    metrics=(
        'RMSE', 'RMSE_ext',
        'R2', 'R2_ext',
        'Bias', 'Bias_ext',
        'VarRatio', 'VarRatio_ext',
        'NSE', 'NSE_ext',
        'KGE', 'KGE_ext',
        'CRPS', 'CRPS_ext'
    ),
    low=5,
    high=95,
    date_col='date'
):
    df_model = df[df['model'] == model].copy()

    if accumulations is None:
        accumulations = [None]
    elif isinstance(accumulations, int):
        accumulations = [accumulations]

    def calc(metric, obs, pred, ens):
        if len(obs) == 0:
            return np.nan

        if metric == 'RMSE':
            return np.sqrt(mean_squared_error(obs, pred))

        if metric == 'R2':
            return r2_score(obs, pred) if len(obs) > 1 else np.nan

        if metric == 'Bias':
            return np.mean(pred - obs)

        if metric == 'VarRatio':
            return np.var(pred) / np.var(obs) if np.var(obs) != 0 else np.nan

        if metric == 'NSE':
            denom = np.sum((obs - np.mean(obs)) ** 2)
            return 1 - np.sum((pred - obs) ** 2) / denom if denom != 0 else np.nan

        if metric == 'KGE':
            r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else np.nan
            alpha = np.std(pred) / np.std(obs) if np.std(obs) != 0 else np.nan
            beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan

            if np.any(np.isnan([r, alpha, beta])):
                return np.nan

            return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        if metric == 'CRPS':
            return np.mean([crps_ensemble(o, p) for o, p in zip(obs, ens)])

        raise ValueError(f"Unknown metric: {metric}")

    all_results = []

    for acc in accumulations:

        metric_store = {m: [] for m in metrics}

        for lake in lakes:
            for comp in components:

                base_var = f"{lake}_{comp}"

                if acc is None:
                    pred_col = base_var
                    obs_col = f"{base_var}_obs"
                else:
                    pred_col = f"{base_var}_acc{acc}"
                    obs_col = f"{base_var}_obs_acc{acc}"

                if pred_col not in df_model.columns or obs_col not in df_model.columns:
                    continue

                data = df_model[[date_col, pred_col, obs_col]].dropna()
                grouped = data.groupby(date_col)

                obs_list, pred_list = [], []

                for _, g in grouped:
                    obs = g[obs_col].iloc[0]
                    preds = g[pred_col].values

                    obs_list.append(obs)
                    pred_list.append(preds)

                if len(obs_list) == 0:
                    continue

                obs_arr = np.array(obs_list)
                pred_mean = np.array([p.mean() for p in pred_list])

                low_thr = np.percentile(obs_arr, low)
                high_thr = np.percentile(obs_arr, high)
                mask = (obs_arr <= low_thr) | (obs_arr >= high_thr)

                obs_ext = obs_arr[mask]
                pred_ext = pred_mean[mask]
                ens_ext = [p for p, m in zip(pred_list, mask) if m]

                for m in metrics:
                    if m.endswith('_ext'):
                        base_metric = m.replace('_ext', '')
                        val = calc(base_metric, obs_ext, pred_ext, ens_ext)
                    else:
                        val = calc(m, obs_arr, pred_mean, pred_list)

                    metric_store[m].append(val)

        results = {
            'Model': model,
            'Accumulation': 'monthly' if acc is None else f'{acc} month'
        }

        for m in metrics:
            vals = np.array(metric_store[m], dtype=float)
            results[m] = np.nanmean(vals) if len(vals) > 0 else np.nan

        all_results.append(results)

    return pd.DataFrame(all_results)