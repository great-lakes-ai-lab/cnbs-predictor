# tests/unit/test_hydro_utils.py
"""
Unit tests for src/hydro_utils.py.

Covers pure calculation functions: known-value checks, unit conversions,
and error-handling guardrails. No external data, no network.
"""

import numpy as np
import pandas as pd
import pytest

from src.hydro_utils import (
    seconds_in_month,
    calculate_grid_cell_areas,
    calculate_evaporation_rate,
    convert_mm_to_cms,
)


# ---------------------------------------------------------------------------
# seconds_in_month
# ---------------------------------------------------------------------------
class TestSecondsInMonth:
    """Tests ``seconds_in_month`` day counts and month-range validation."""

    def test_january_has_31_days(self):
        """January returns 31 days' worth of seconds."""
        assert seconds_in_month(2023, 1) == 31 * 86400

    def test_february_non_leap(self):
        """February in a non-leap year returns 28 days' worth of seconds."""
        assert seconds_in_month(2023, 2) == 28 * 86400

    def test_february_leap(self):
        """February in a leap year returns 29 days' worth of seconds."""
        assert seconds_in_month(2024, 2) == 29 * 86400

    def test_april_has_30_days(self):
        """April returns 30 days' worth of seconds."""
        assert seconds_in_month(2023, 4) == 30 * 86400

    @pytest.mark.parametrize("bad_month", [0, 13, -1, 100])
    def test_invalid_month_raises(self, bad_month):
        """A month outside 1-12 raises ValueError."""
        with pytest.raises(ValueError, match="Month must be between 1 and 12"):
            seconds_in_month(2023, bad_month)


# ---------------------------------------------------------------------------
# calculate_grid_cell_areas
# ---------------------------------------------------------------------------
class TestCalculateGridCellAreas:
    """Tests grid-cell area computation by ``calculate_grid_cell_areas``."""

    def test_returns_2d_array_with_correct_shape(self):
        """Returns a 2D (n_lat, n_lon) array."""
        lon = np.array([0.0, 1.0, 2.0])
        lat = np.array([10.0, 11.0])
        area = calculate_grid_cell_areas(lon, lat)
        assert area.shape == (2, 3)

    def test_areas_are_positive(self):
        """All computed cell areas are positive."""
        lon = np.linspace(-90, -75, 5)
        lat = np.linspace(40, 50, 4)
        area = calculate_grid_cell_areas(lon, lat)
        assert (area > 0).all()

    def test_area_decreases_toward_pole(self):
        """A cell at higher latitude should be smaller than one nearer the equator."""
        lon = np.array([0.0, 1.0])
        lat = np.array([0.0, 80.0])
        area = calculate_grid_cell_areas(lon, lat)
        # row 0 = equator, row 1 = high latitude
        assert area[0, 0] > area[1, 0]

    def test_known_value_at_equator(self):
        """At the equator, area ≈ R^2 * dlat * dlon * cos(0) = R^2 * dlat * dlon."""
        lon = np.array([0.0, 1.0])
        lat = np.array([0.0, 1.0])
        area = calculate_grid_cell_areas(lon, lat)
        R = 6371000.0
        expected = R**2 * np.radians(1.0) * np.radians(1.0) * np.cos(0.0)
        assert area[0, 0] == pytest.approx(expected, rel=1e-12)

    def test_2d_input_raises(self):
        """Raises ValueError when given 2D lon/lat arrays instead of 1D."""
        lon = np.zeros((2, 2))
        lat = np.zeros((2, 2))
        with pytest.raises(ValueError, match="must be 1D arrays"):
            calculate_grid_cell_areas(lon, lat)


# ---------------------------------------------------------------------------
# calculate_evaporation_rate
# ---------------------------------------------------------------------------
class TestCalculateEvaporation:
    """Tests ``calculate_evaporation_rate`` formula, dtype handling, and monotonicity."""

    def test_scalar_inputs_return_scalar(self):
        """Scalar inputs return the expected scalar evaporation rate."""
        result = calculate_evaporation_rate(temperature_K=288.15, latent_heat_flux=2_500_000.0)
        # latent_heat W/m² * 1e-6 / lambda(MJ/kg)  -> kg/(m²·s)
        # at 15°C: lambda = 2.501 - 0.002361*15 = 2.46557
        expected = (2_500_000.0 * 1e-6) / (2.501 - 0.002361 * 15.0)
        assert result == pytest.approx(expected, rel=1e-9)

    def test_array_inputs_return_array(self):
        """Array inputs return a positive array of matching shape."""
        T = np.array([273.15, 288.15, 303.15])
        LE = np.array([1e6, 2e6, 3e6])
        result = calculate_evaporation_rate(T, LE)
        assert isinstance(result, np.ndarray)
        assert result.shape == T.shape
        assert (result > 0).all()

    def test_zero_latent_heat_gives_zero_evaporation(self):
        """Zero latent-heat flux yields zero evaporation."""
        assert calculate_evaporation_rate(288.15, 0.0) == 0.0

    def test_higher_latent_heat_means_more_evaporation(self):
        """Higher latent-heat flux yields more evaporation at the same temperature."""
        low = calculate_evaporation_rate(288.15, 1e6)
        high = calculate_evaporation_rate(288.15, 2e6)
        assert high > low


# ---------------------------------------------------------------------------
# convert_mm_to_cms
# ---------------------------------------------------------------------------
class TestConvertMmToCms:
    """Tests mm-to-cms conversion (per-lake area + month length) by ``convert_mm_to_cms``."""

    def _toy_df(self):
        """Build a four-lake toy DataFrame of 10 mm values for January 2023."""
        return pd.DataFrame(
            {
                "value [mm]": [10.0, 10.0, 10.0, 10.0],
                "lake": ["superior", "michigan-huron", "erie", "ontario"],
                "year": [2023, 2023, 2023, 2023],
                "month": [1, 1, 1, 1],
            }
        )

    def test_adds_cms_column(self):
        """Adds a 'value [cms]' column to the output."""
        df = convert_mm_to_cms(self._toy_df())
        assert "value [cms]" in df.columns

    def test_known_value_superior_january(self):
        """10 mm over Superior (82,097 km²) over Jan (31 days) — verify exact formula."""
        df = convert_mm_to_cms(self._toy_df())
        sup_sa_m2 = 82097 * 1_000_000
        secs = 31 * 86400
        expected = (10.0 / 1000.0) * sup_sa_m2 / secs
        sup_row = df.loc[df["lake"] == "superior", "value [cms]"].iloc[0]
        assert sup_row == pytest.approx(expected, rel=1e-12)

    def test_michigan_huron_uses_combined_area(self):
        """Michigan-Huron conversion uses the combined Michigan + Huron surface area."""
        df = convert_mm_to_cms(self._toy_df())
        combined_sa = (57753 + 59560) * 1_000_000
        secs = 31 * 86400
        expected = (10.0 / 1000.0) * combined_sa / secs
        mh = df.loc[df["lake"] == "michigan-huron", "value [cms]"].iloc[0]
        assert mh == pytest.approx(expected, rel=1e-12)

    def test_unknown_lake_yields_zero(self):
        """An unrecognized lake name converts to a cms value of zero."""
        df = pd.DataFrame(
            {
                "value [mm]": [10.0],
                "lake": ["loch_ness"],
                "year": [2023],
                "month": [1],
            }
        )
        out = convert_mm_to_cms(df)
        assert out["value [cms]"].iloc[0] == 0.0

    def test_proportional_to_mm(self):
        """Doubling mm should double cms (same lake/month)."""
        df = pd.DataFrame(
            {
                "value [mm]": [5.0, 10.0],
                "lake": ["erie", "erie"],
                "year": [2023, 2023],
                "month": [6, 6],
            }
        )
        out = convert_mm_to_cms(df)
        assert out["value [cms]"].iloc[1] == pytest.approx(
            2.0 * out["value [cms]"].iloc[0], rel=1e-12
        )
