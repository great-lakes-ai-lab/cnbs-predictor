# tests/test_seasonal_cycle_processor.py

import numpy as np
import pandas as pd
import pytest
from src.data_processor import SeasonalCycleProcessor


def _toy_monthly_df():
    """
    Create a tiny monthly DataFrame with a DatetimeIndex.
    Values are constructed so that different months have different means.
    """
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")  # Jan 2000 .. Dec 2001
    # Make var_a depend on month strongly; var_b is constant-ish + month
    month = idx.month
    df = pd.DataFrame(
        {
            "var_a": 100.0 + month,      # Jan=101, Feb=102, ... Dec=112 (repeats each year)
            "var_b": 5.0 + 2.0 * month,  # Jan=7, ... Dec=29
        },
        index=idx,
    )
    df.index.name = "date"
    return df


def test_roundtrip_transform_inverse_transform():
    """
    Acceptance criterion:
    inverse_transform(transform(x)) == x (within float tolerance)
    """
    df = _toy_monthly_df()

    scp = SeasonalCycleProcessor().fit(
        df,
        var_list=["var_a", "var_b"],
        baseline_time=slice("2000-01-01", "2000-12-01"),
        baseline_definition={"note": "baseline is year 2000 only"},
    )

    anom = scp.transform(df)
    recon = scp.inverse_transform(anom)

    pd.testing.assert_frame_equal(recon, df, check_exact=False, rtol=1e-12, atol=1e-10)


def test_month_indexing_jan_vs_dec_anomalies():
    """
    Month indexing check:
    - If climatology for Jan differs from Dec, anomalies should reflect that.
    - Also covers edge months (1 and 12).
    """
    df = _toy_monthly_df()

    # Fit on the first year only
    scp = SeasonalCycleProcessor().fit(
        df,
        var_list=["var_a", "var_b"],
        baseline_time=slice("2000-01-01", "2000-12-01"),
        baseline_definition={"note": "baseline is year 2000 only"},
    )

    # In baseline year, since each month appears once, climatology for month m equals raw value in that month.
    # Therefore anomalies for year 2000 should be exactly zero.
    baseline = df.loc["2000-01-01":"2000-12-01"]
    baseline_anom = scp.transform(baseline)
    assert np.allclose(baseline_anom[["var_a", "var_b"]].to_numpy(), 0.0)

    # For year 2001, values repeat exactly (same construction), so anomalies should also be zero.
    yr2001 = df.loc["2001-01-01":"2001-12-01"]
    yr2001_anom = scp.transform(yr2001)
    assert np.allclose(yr2001_anom[["var_a", "var_b"]].to_numpy(), 0.0)

    # Explicitly check January and December rows exist and were handled
    jan = scp.transform(df.loc[["2001-01-01"]])
    dec = scp.transform(df.loc[["2001-12-01"]])
    assert jan.index[0].month == 1
    assert dec.index[0].month == 12


def test_fit_requires_datetime_index():
    """
    Guardrail: processor should raise if DataFrame lacks a DatetimeIndex.
    """
    df = pd.DataFrame({"var_a": [1.0, 2.0], "var_b": [3.0, 4.0]})
    scp = SeasonalCycleProcessor()
    with pytest.raises(ValueError):
        scp.fit(df)


def test_transform_requires_fit():
    """
    Guardrail: transform before fit should raise.
    """
    df = _toy_monthly_df()
    scp = SeasonalCycleProcessor()
    with pytest.raises(ValueError):
        scp.transform(df)