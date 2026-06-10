# tests/unit/test_transformers.py
"""
Unit tests for ``CFSTransformer`` and ``ForecastTransformer`` in
``src/data_processor.py``.

These transformers reshape long/wide DataFrames into the exact column layouts
the downstream models expect. The whole point of the classes is producing a
specific column order, so these tests pin that order with synthetic inputs —
no real data files needed.
"""

from datetime import datetime
import itertools

import pandas as pd
import pytest

from src.data_processor import CFSTransformer, ForecastTransformer


# ===========================================================================
# Constructor validation (both transformers)
# ===========================================================================
class TestConstructorValidation:
    @pytest.mark.parametrize("bad_input", [None, 42, "not a df", [1, 2, 3]])
    def test_cfs_transformer_rejects_non_dataframe(self, bad_input):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            CFSTransformer(bad_input)

    @pytest.mark.parametrize("bad_input", [None, 42, "not a df", [1, 2, 3]])
    def test_forecast_transformer_rejects_non_dataframe(self, bad_input):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            ForecastTransformer(bad_input)


# ===========================================================================
# CFSTransformer.filter
# ===========================================================================
class TestCFSTransformerFilter:
    """
    ``filter(first_forecast_month, months_back=10)`` keeps rows whose
    ``cfs_run`` is ≥ ``first_forecast_month - months_back months``.
    """

    @staticmethod
    def _build(cfs_runs):
        return pd.DataFrame({"cfs_run": cfs_runs, "value": range(len(cfs_runs))})

    def test_filters_out_runs_before_window(self):
        df = self._build([
            "2024010100",  # before window — drop
            "2024020100",  # within — keep
            "2024030100",  # within — keep
            "2024040100",  # within — keep
            "2024110100",  # within — keep
            "2024120100",  # equal to first_fc — keep
        ])
        result = CFSTransformer(df).filter(datetime(2024, 12, 1), months_back=10)
        print(result)
        # Three rows survive; the 2024-01 one is dropped.
        assert len(result) == 5
        assert "2024010100" not in result["cfs_run"].astype(str).values

    def test_default_months_back_is_10(self):
        df = self._build(["2023060100", "2024030100", "2024120100"])
        # Default months_back=10, first_fc=2024-12 -> start = 2024-03-01.
        result = CFSTransformer(df).filter(datetime(2024, 12, 1))
        assert len(result) == 2  # only going back 9 months, so 2023-06-01 and 2024-03-01 are not included


# ===========================================================================
# CFSTransformer.shift_variables
# ===========================================================================
class TestCFSTransformerShift:
    """
    ``shift_variables(lag, lead)`` adds shifted columns and renames the
    originals to ``_mo0``. Note: the production code uses ``range(1, lag)``,
    so passing ``lag=2`` actually creates only one lag column (mo-1). This
    test pins the current behavior.
    """

    def test_lag_zero_lead_zero_only_renames_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        out = CFSTransformer(df).shift_variables(lag=0, lead=0)
        assert list(out.columns) == ["a_mo0", "b_mo0"]
        assert len(out) == 3

    def test_lag_2_creates_one_lag_column_per_var(self):
        """Sub-optimal: range(1, lag) means lag=2 -> only mo-1 column. Pinned."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        out = CFSTransformer(df).shift_variables(lag=2, lead=0)
        assert "a_mo0" in out.columns
        assert "a_mo-1" in out.columns
        # Rows with NaN from shifting are dropped.
        assert len(out) == 3

    def test_dropna_removes_rows_with_shift_induced_nan(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = CFSTransformer(df).shift_variables(lag=3, lead=3)
        # Lags up to mo-2 and leads up to mo2 → first 2 and last 2 rows have NaN → dropped.
        assert len(out) == 1


# ===========================================================================
# CFSTransformer.structure_input — the column-order contract
# ===========================================================================
class TestCFSTransformerStructureInput:
    """
    structure_input produces a wide DataFrame with:

      - 240 feature columns:
        4 lakes × 2 surface_types × 3 components × 10 forecast months

    Total: 252 columns in deterministic order.
    """

    LAKES = ["superior", "michigan-huron", "erie", "ontario"]
    SURFACES = ["lake", "land"]
    COMPONENTS = ["precipitation", "evaporation", "air_temperature"]

    def _build_long_input(self, cfs_run="2024010100"):
        cfs_dt = pd.to_datetime(cfs_run, format="%Y%m%d%H")

        rows = []
        for lake, surface, comp, mo in itertools.product(
            self.LAKES, self.SURFACES, self.COMPONENTS, range(11)
        ):
            forecast_date = cfs_dt + pd.DateOffset(months=mo)
            rows.append({
                "cfs_run": cfs_run,
                "year": forecast_date.year,
                "month": forecast_date.month,
                "lake": lake,
                "surface_type": surface,
                "component": comp,
                "value": 1.0 + mo,
            })

        return pd.DataFrame(rows)

    # -----------------------------
    # Tests
    # -----------------------------

    def test_returns_dataframe_with_expected_total_columns(self):
        df = self._build_long_input()
        result = CFSTransformer(df).structure_input()

        assert result.shape[1] == 240 # 4 lakes × 2 surfaces × 3 components × 10 months

    def test_feature_columns_in_nested_lake_surface_component_month_order(self):
        df = self._build_long_input()
        result = CFSTransformer(df).structure_input()

        feature_cols = list(result.columns)

        expected = [
            f"{lake}_{surface_type}_{comp}_mo{m}"
            for lake in ["superior", "michigan-huron", "erie", "ontario"]
            for surface_type in ["lake", "land"]
            for comp in ["precipitation", "evaporation", "air_temperature"]
            for m in range(10)
        ]

        assert feature_cols == expected

    def test_no_mo10_columns_in_output(self):
        df = self._build_long_input()
        result = CFSTransformer(df).structure_input()

        assert not any(c.endswith("_mo10") for c in result.columns)

    def test_accepts_value_mm_column_alias(self):
        df = self._build_long_input().rename(columns={"value": "value [mm]"})
        result = CFSTransformer(df).structure_input()

        assert result.shape[1] == 240  # Still produces 240 feature columns

import numpy as np
import pandas as pd
import pytest


class TestAddTimeFeatures:

    def _build_df(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="MS")
        return pd.DataFrame({"value": [1, 2, 3, 4]}, index=idx)

    # ---------------------------------------------------------
    # Raises error if index is not DatetimeIndex
    # ---------------------------------------------------------
    def test_requires_datetime_index(self):
        df = pd.DataFrame({"value": [1, 2, 3]})

        with pytest.raises(TypeError, match="DatetimeIndex"):
            CFSTransformer(df).add_time_features()

    # ---------------------------------------------------------
    # Adds month_sin and month_cos correctly
    # ---------------------------------------------------------
    def test_month_cycle_features(self):
        df = self._build_df()

        result = CFSTransformer(df).add_time_features(
            add_time_trend=False
        )

        # expected values for Jan, Feb, Mar, Apr
        months = np.array([1, 2, 3, 4])

        expected_sin = np.sin(2 * np.pi * months / 12)
        expected_cos = np.cos(2 * np.pi * months / 12)

        np.testing.assert_allclose(result["month_sin"].values, expected_sin)
        np.testing.assert_allclose(result["month_cos"].values, expected_cos)


    # ---------------------------------------------------------
    # Time normalization centers data
    # ---------------------------------------------------------
    def test_time_trend_normalized(self):
        df = self._build_df()

        result = CFSTransformer(df).add_time_features(
            add_month_cycle=False,
            normalize_time=True
        )

        time = result["time"].values

        # mean ~ 0
        assert abs(time.mean()) < 1e-10

        # std ~ 1 (unless constant series)
        assert abs(time.std() - 1) < 1e-10

    # ---------------------------------------------------------
    # Overwrite removes old columns
    # ---------------------------------------------------------
    def test_overwrite_behavior(self):
        df = self._build_df()
        df["month_sin"] = 999  # fake old value

        result = CFSTransformer(df).add_time_features(overwrite=True)

        # overwritten
        assert result["month_sin"].iloc[0] != 999

# ===========================================================================
# ForecastTransformer.melt — wide-format → tidy with sorted month columns
# ===========================================================================
class TestForecastTransformerMelt:
    """
    ``melt`` takes a wide DataFrame with columns like
    ``<lake>_<component>_mo<N>`` and reshapes to one row per (cfs_run, model,
    lake, component) with sorted month columns.
    """

    def _build_wide_input(self):
        return pd.DataFrame([{
            "cfs_run": "2024010100",
            "model": "GP",
            "superior_precipitation_mo0":  10.0,
            "superior_precipitation_mo1":  20.0,
            "superior_precipitation_mo2":  30.0,
            "superior_evaporation_mo0":    1.0,
            "superior_evaporation_mo1":    2.0,
            "superior_evaporation_mo2":    3.0,
        }])

    def test_id_vars_come_first(self):
        df = self._build_wide_input()
        out = ForecastTransformer(df).melt()
        assert list(out.columns[:4]) == ["cfs_run", "model", "lake", "component"]

    def test_month_columns_sorted_numerically_not_lexically(self):
        """
        The post-pivot month columns are named ``month_0``, ``month_1``, ...
        Lexical sort would put ``month_10`` before ``month_2`` — the loader
        sorts numerically. This test guards that contract.
        """
        # Build wide input with mo0..mo11 to force the lexical-vs-numeric distinction.
        wide = {"cfs_run": "2024010100", "model": "GP"}
        for m in range(12):
            wide[f"superior_precipitation_mo{m}"] = float(m)
        df = pd.DataFrame([wide])
        out = ForecastTransformer(df).melt()
        month_cols = [c for c in out.columns if c.startswith("month_")]
        # Numeric order, not lexical.
        assert month_cols == [f"month_{i}" for i in range(12)]

    def test_melts_handles_michigan_huron_with_dash(self):
        """The lake split uses rsplit so 'michigan-huron' stays intact."""
        df = pd.DataFrame([{
            "cfs_run": "2024010100",
            "model": "GP",
            "michigan-huron_precipitation_mo0": 5.0,
            "michigan-huron_precipitation_mo1": 6.0,
        }])
        out = ForecastTransformer(df).melt()
        assert "michigan-huron" in out["lake"].values


# ===========================================================================
# ForecastTransformer.pivot — long format with calculated forecast_month
# ===========================================================================
class TestForecastTransformerPivot:
    """
    ``pivot`` reshapes so each variable type becomes its own column. The
    final column order is:
        ['cfs_run', 'forecast_month', 'model', 'lake', <variables in known order>]
    where the known variable order is precipitation, evaporation, runoff, nbs.
    """

    def _build_input(self):
        return pd.DataFrame([{
            "cfs_run": "2024010100",
            "model": "GP",
            "superior_precipitation_mo0": 100.0,
            "superior_precipitation_mo1": 110.0,
            "superior_evaporation_mo0":   10.0,
            "superior_evaporation_mo1":   11.0,
        }])

    def test_id_columns_come_first_then_variables(self):
        df = self._build_input()
        out = ForecastTransformer(df).pivot()
        assert list(out.columns[:4]) == ["cfs_run", "forecast_month", "model", "lake"]

    def test_variables_in_canonical_order(self):
        """
        Per the loader, the canonical order is precipitation, evaporation,
        runoff, nbs — only those that exist in the input.
        """
        df = self._build_input()
        out = ForecastTransformer(df).pivot()
        var_cols = [c for c in out.columns if c not in
                    {"cfs_run", "forecast_month", "model", "lake"}]
        assert var_cols == ["precipitation", "evaporation"]

    def test_cfs_run_string_format_preserved(self):
        df = self._build_input()
        out = ForecastTransformer(df).pivot()
        # cfs_run is round-tripped to "%Y%m%d%H" string.
        assert out["cfs_run"].iloc[0] == "2024010100"

    def test_forecast_month_calculated_from_offset(self):
        """
        ``mo0`` corresponds to the cfs_run month, ``mo1`` to next month, etc.
        For cfs_run='2024010100' (Jan 2024), mo0 → '2024-01', mo1 → '2024-02'.
        """
        df = self._build_input()
        out = ForecastTransformer(df).pivot()
        forecast_months = set(out["forecast_month"])
        assert {"2024-01", "2024-02"} == forecast_months
