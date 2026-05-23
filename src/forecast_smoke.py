"""
Smoke checks for CNBS forecast output dataframes.

Lightweight sanity checks that the forecast pipeline produced an output
that is structurally correct and within plausible ranges. Distinct from
the team's skill-validation work (RMSE/CRPS/R^2 against the forecast
archive) — this only confirms the machinery ran and produced a sensible
shape; it does not assess forecast quality.

The forecast pipeline (notebooks/production/2_LEF_forecast_model.ipynb)
produces a tidy DataFrame with one row per (cfs_run, forecast_month, model,
lake) and columns for each component (precipitation, evaporation, runoff,
nbs). These checks assert that output matches the expected contract before
it is written to disk or database.

Failures raise ValueError with a message describing the offending rows /
columns, so a regression in the upstream pipeline (sign flip, unit bug,
schema drift, NaN leak) fails loudly at the boundary rather than silently
corrupting downstream artifacts.

Typical use, from notebook 2 right before the to_csv / to_sql calls:

    from src.forecast_smoke import smoke_check_forecast
    smoke_check_forecast(df_pivoted)

Tests can call individual checks (smoke_check_schema, smoke_check_no_nans,
smoke_check_ranges) for finer-grained assertions.
"""

import pandas as pd

EXPECTED_LAKES = frozenset({"superior", "michigan-huron", "erie", "ontario"})
EXPECTED_COMPONENTS = ("precipitation", "evaporation", "runoff", "nbs")
ID_COLUMNS = ("cfs_run", "forecast_month", "model", "lake")
REQUIRED_COLUMNS = ID_COLUMNS + EXPECTED_COMPONENTS

# Plausible ranges in mm/month. Generous envelopes informed by GLCC/L2SWBM
# observed climatology — wide enough to never trip on a real run, tight
# enough to catch sign flips, unit errors, or exploded predictions.
COMPONENT_RANGES_MM = {
    "precipitation": (0.0, 500.0),
    "evaporation": (-100.0, 500.0),
    "runoff": (0.0, 500.0),
    "nbs": (-500.0, 800.0),
}

# Sentinel fill value used in some upstream observation files (GLCC).
# Should never appear in model output; if it does, NaN handling broke.
FILL_VALUES = (-99990.0, -9999.0)


def smoke_check_schema(df):
    """
    Check that the forecast dataframe has the expected columns, lake names,
    and forecast_month / cfs_run formats.

    Raises
    ------
    ValueError
        If required columns are missing, lake names are wrong, or time
        index strings don't match the documented format.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas DataFrame, got {type(df).__name__}")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Forecast dataframe is missing required columns: {sorted(missing)}"
        )

    if len(df) == 0:
        raise ValueError("Forecast dataframe is empty")

    unexpected_lakes = set(df["lake"].unique()) - EXPECTED_LAKES
    if unexpected_lakes:
        raise ValueError(
            f"Forecast contains unexpected lake names: {sorted(unexpected_lakes)}; "
            f"expected subset of {sorted(EXPECTED_LAKES)}"
        )

    # cfs_run: 10-digit string YYYYMMDDHH
    bad_runs = df.loc[~df["cfs_run"].astype(str).str.fullmatch(r"\d{10}"), "cfs_run"]
    if not bad_runs.empty:
        raise ValueError(
            f"cfs_run values must be 10-digit YYYYMMDDHH strings; "
            f"first offender: {bad_runs.iloc[0]!r}"
        )

    # forecast_month: YYYY-MM
    bad_months = df.loc[
        ~df["forecast_month"].astype(str).str.fullmatch(r"\d{4}-\d{2}"),
        "forecast_month",
    ]
    if not bad_months.empty:
        raise ValueError(
            f"forecast_month values must be YYYY-MM strings; "
            f"first offender: {bad_months.iloc[0]!r}"
        )


def smoke_check_no_nans(df):
    """
    Check that no NaN or sentinel fill values appear in id columns or
    component columns.

    Raises
    ------
    ValueError
        If any required column contains NaN, or if any component column
        contains a known sentinel fill value (-99990.0 / -9999.0).
    """
    cols_to_check = [c for c in REQUIRED_COLUMNS if c in df.columns]
    nan_counts = df[cols_to_check].isna().sum()
    offenders = nan_counts[nan_counts > 0]
    if not offenders.empty:
        raise ValueError(
            f"Forecast contains NaN values in: {offenders.to_dict()}"
        )

    component_cols = [c for c in EXPECTED_COMPONENTS if c in df.columns]
    for col in component_cols:
        for fill in FILL_VALUES:
            n = (df[col] == fill).sum()
            if n > 0:
                raise ValueError(
                    f"Forecast column '{col}' contains {n} sentinel fill "
                    f"value(s) ({fill}); upstream NaN handling likely broke"
                )


def smoke_check_ranges(df, ranges=None):
    """
    Check that each component value falls within a plausible mm/month range.

    Parameters
    ----------
    df : pandas.DataFrame
        Forecast output; expected to have columns from EXPECTED_COMPONENTS.
    ranges : dict, optional
        Override the default per-component (low, high) bounds. Defaults to
        COMPONENT_RANGES_MM.

    Raises
    ------
    ValueError
        If any component value falls outside its bounds. The message
        identifies the column, bound, and an example offending row.
    """
    bounds = ranges if ranges is not None else COMPONENT_RANGES_MM

    for col, (low, high) in bounds.items():
        if col not in df.columns:
            continue
        vals = df[col]
        below = vals < low
        above = vals > high
        if below.any():
            row = df.loc[below].iloc[0]
            raise ValueError(
                f"Forecast column '{col}' has values below {low} mm/month; "
                f"first offender: lake={row.get('lake')!r}, "
                f"forecast_month={row.get('forecast_month')!r}, "
                f"value={vals.loc[below].iloc[0]}"
            )
        if above.any():
            row = df.loc[above].iloc[0]
            raise ValueError(
                f"Forecast column '{col}' has values above {high} mm/month; "
                f"first offender: lake={row.get('lake')!r}, "
                f"forecast_month={row.get('forecast_month')!r}, "
                f"value={vals.loc[above].iloc[0]}"
            )


def smoke_check_forecast(df, ranges=None):
    """
    Run all forecast output smoke checks: schema, no-NaNs, value ranges.

    Call this from notebook 2 just before writing CSVs / to_sql.

    Raises
    ------
    ValueError
        On the first failed check, with a message describing the failure.
    """
    smoke_check_schema(df)
    smoke_check_no_nans(df)
    smoke_check_ranges(df, ranges=ranges)
