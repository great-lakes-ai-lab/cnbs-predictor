"""
Unit tests for src/forecast_smoke.py.

Exercises each smoke check in two directions:
- a healthy forecast (the shipped fixture, plus synthetic dataframes) passes,
- a deliberately broken forecast raises ValueError with a useful message.

The shipped fixture (tests/fixtures/forecasts/CNBS_forecast_wide.tsv) is a
small hand-built sample matching the schema produced by
ForecastTransformer.pivot() in notebook 2. Values come from the observed
monthly climatology (data/input/climatology/targets/climatology.csv) so a
real run would land in the same envelope.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast_smoke import (
    COMPONENT_RANGES_MM,
    EXPECTED_COMPONENTS,
    EXPECTED_LAKES,
    REQUIRED_COLUMNS,
    smoke_check_forecast,
    smoke_check_no_nans,
    smoke_check_ranges,
    smoke_check_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "forecasts" / "CNBS_forecast_wide.tsv"


def _load_fixture():
    return pd.read_csv(FIXTURE, sep="\t", dtype={"cfs_run": str, "forecast_month": str})


def _toy_forecast():
    """Build a minimal in-memory forecast covering all four lakes."""
    rows = []
    for lake in EXPECTED_LAKES:
        rows.append({
            "cfs_run": "2024090400",
            "forecast_month": "2024-09",
            "model": "GP",
            "lake": lake,
            "precipitation": 80.0,
            "evaporation": 50.0,
            "runoff": 40.0,
            "nbs": 70.0,
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Fixture loads cleanly and all checks pass on it
# ===========================================================================
class TestFixturePasses:
    def test_fixture_file_exists(self):
        assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"

    def test_smoke_check_forecast_passes_on_fixture(self):
        smoke_check_forecast(_load_fixture())

    def test_smoke_check_forecast_passes_on_toy(self):
        smoke_check_forecast(_toy_forecast())


# ===========================================================================
# smoke_check_schema
# ===========================================================================
class TestSmokeCheckSchema:
    def test_passes_on_healthy_dataframe(self):
        smoke_check_schema(_load_fixture())

    def test_rejects_non_dataframe(self):
        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            smoke_check_schema({"not": "a df"})

    def test_rejects_empty_dataframe(self):
        empty = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        with pytest.raises(ValueError, match="empty"):
            smoke_check_schema(empty)

    def test_rejects_missing_component_column(self):
        df = _load_fixture().drop(columns=["nbs"])
        with pytest.raises(ValueError, match="missing required columns"):
            smoke_check_schema(df)

    def test_rejects_missing_id_column(self):
        df = _load_fixture().drop(columns=["forecast_month"])
        with pytest.raises(ValueError, match="missing required columns"):
            smoke_check_schema(df)

    def test_rejects_unexpected_lake_name(self):
        df = _load_fixture()
        df.loc[0, "lake"] = "saint-clair"
        with pytest.raises(ValueError, match="unexpected lake names"):
            smoke_check_schema(df)

    def test_rejects_bad_cfs_run_format(self):
        df = _load_fixture()
        df.loc[0, "cfs_run"] = "20240904"  # 8 digits, not 10
        with pytest.raises(ValueError, match="cfs_run"):
            smoke_check_schema(df)

    def test_rejects_bad_forecast_month_format(self):
        df = _load_fixture()
        df.loc[0, "forecast_month"] = "Sep 2024"
        with pytest.raises(ValueError, match="forecast_month"):
            smoke_check_schema(df)

    def test_subset_of_lakes_is_allowed(self):
        """The fixture has all four lakes, but a single-lake forecast should
        still pass schema (subset of EXPECTED_LAKES is fine)."""
        df = _load_fixture()
        df_one_lake = df[df["lake"] == "superior"].reset_index(drop=True)
        smoke_check_schema(df_one_lake)


# ===========================================================================
# smoke_check_no_nans
# ===========================================================================
class TestSmokeCheckNoNans:
    def test_passes_on_healthy_dataframe(self):
        smoke_check_no_nans(_load_fixture())

    def test_rejects_nan_in_component_column(self):
        df = _load_fixture()
        df.loc[0, "precipitation"] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            smoke_check_no_nans(df)

    def test_rejects_nan_in_id_column(self):
        df = _load_fixture()
        df.loc[0, "forecast_month"] = np.nan
        with pytest.raises(ValueError, match="NaN values"):
            smoke_check_no_nans(df)

    def test_rejects_glcc_sentinel_fill_value(self):
        df = _load_fixture()
        df.loc[0, "nbs"] = -99990.0
        with pytest.raises(ValueError, match="sentinel fill"):
            smoke_check_no_nans(df)

    def test_rejects_alt_sentinel_fill_value(self):
        df = _load_fixture()
        df.loc[0, "evaporation"] = -9999.0
        with pytest.raises(ValueError, match="sentinel fill"):
            smoke_check_no_nans(df)


# ===========================================================================
# smoke_check_ranges
# ===========================================================================
class TestSmokeCheckRanges:
    def test_passes_on_healthy_dataframe(self):
        smoke_check_ranges(_load_fixture())

    def test_rejects_negative_precipitation(self):
        """Sign flip: precipitation is mass, can't be negative."""
        df = _load_fixture()
        df.loc[0, "precipitation"] = -10.0
        with pytest.raises(ValueError, match="precipitation.*below"):
            smoke_check_ranges(df)

    def test_rejects_negative_runoff(self):
        df = _load_fixture()
        df.loc[0, "runoff"] = -5.0
        with pytest.raises(ValueError, match="runoff.*below"):
            smoke_check_ranges(df)

    def test_rejects_implausibly_negative_evaporation(self):
        """Evaporation can be slightly negative, but not catastrophically."""
        df = _load_fixture()
        df.loc[0, "evaporation"] = -500.0
        with pytest.raises(ValueError, match="evaporation.*below"):
            smoke_check_ranges(df)

    def test_allows_slightly_negative_evaporation(self):
        """L2SWBM evap can be slightly negative on a real dataset."""
        df = _load_fixture()
        df.loc[0, "evaporation"] = -50.0
        smoke_check_ranges(df)

    def test_rejects_exploded_value_above_upper_bound(self):
        """Catches a unit error / blown-up prediction."""
        df = _load_fixture()
        df.loc[0, "precipitation"] = 99999.0
        with pytest.raises(ValueError, match="precipitation.*above"):
            smoke_check_ranges(df)

    def test_rejects_implausibly_large_negative_nbs(self):
        df = _load_fixture()
        df.loc[0, "nbs"] = -2000.0
        with pytest.raises(ValueError, match="nbs.*below"):
            smoke_check_ranges(df)

    def test_rejects_implausibly_large_positive_nbs(self):
        df = _load_fixture()
        df.loc[0, "nbs"] = 5000.0
        with pytest.raises(ValueError, match="nbs.*above"):
            smoke_check_ranges(df)

    def test_custom_ranges_override_defaults(self):
        """Caller can pass tighter bounds for a more aggressive check."""
        df = _load_fixture()
        tight = {"precipitation": (0.0, 50.0)}  # fixture has values up to ~95
        with pytest.raises(ValueError, match="precipitation.*above"):
            smoke_check_ranges(df, ranges=tight)

    def test_error_message_identifies_offender(self):
        """The error should name the lake and forecast_month so the failure
        is debuggable from logs alone."""
        df = _load_fixture()
        df.loc[0, "precipitation"] = -10.0
        with pytest.raises(ValueError) as excinfo:
            smoke_check_ranges(df)
        msg = str(excinfo.value)
        assert df.loc[0, "lake"] in msg
        assert df.loc[0, "forecast_month"] in msg


# ===========================================================================
# smoke_check_forecast composes the others
# ===========================================================================
class TestSmokeCheckForecast:
    def test_passes_on_fixture(self):
        smoke_check_forecast(_load_fixture())

    def test_fails_on_schema_break(self):
        df = _load_fixture().drop(columns=["nbs"])
        with pytest.raises(ValueError, match="missing required columns"):
            smoke_check_forecast(df)

    def test_fails_on_nan(self):
        df = _load_fixture()
        df.loc[0, "precipitation"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            smoke_check_forecast(df)

    def test_fails_on_range_break(self):
        df = _load_fixture()
        df.loc[0, "precipitation"] = -10.0
        with pytest.raises(ValueError, match="below"):
            smoke_check_forecast(df)


# ===========================================================================
# Defaults are sane (regression guard against accidental edits to constants)
# ===========================================================================
class TestModuleConstants:
    def test_expected_components_match_required_columns(self):
        for component in EXPECTED_COMPONENTS:
            assert component in REQUIRED_COLUMNS

    def test_every_component_has_a_range(self):
        for component in EXPECTED_COMPONENTS:
            assert component in COMPONENT_RANGES_MM
            low, high = COMPONENT_RANGES_MM[component]
            assert low < high, f"{component} has inverted bounds"

    def test_precipitation_lower_bound_is_non_negative(self):
        """Mass-balance: precip can't be negative."""
        assert COMPONENT_RANGES_MM["precipitation"][0] >= 0

    def test_runoff_lower_bound_is_non_negative(self):
        assert COMPONENT_RANGES_MM["runoff"][0] >= 0
