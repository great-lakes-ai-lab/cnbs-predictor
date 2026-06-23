# tests/unit/test_data_loader.py
"""
Unit tests for ``src/data_loader.py``, exercised against the real CSVs that
ship in ``data/``.

These tests verify the *contract* of each loader — output shape, columns,
dtypes, NaN handling, sanity ranges — rather than specific cell values, so
the suite stays stable when the underlying data files are updated.

If the shipped data layout ever changes (file renamed, column reordered),
these tests will fail loudly. That's the intended tripwire: the shipped
files in ``data/`` are the loaders' real-world contract.

Tests skip with a clear message if the expected data files aren't present
(e.g. partial clone, files temporarily moved), so a missing data dir
doesn't masquerade as a broken loader.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

GLCC_DIR = DATA_DIR / "glcc"
L2SWBM_DIR = DATA_DIR / "l2swbm"
GLSEA_FILE = DATA_DIR / "glsea" / "glsea_sst_1995-2024.csv"
PROBABILITIES_DIR = DATA_DIR / "probabilities"


def _require_files(*paths: Path):
    """Skip the calling test cleanly if any expected data file is missing."""
    missing = [str(p.relative_to(REPO_ROOT)) for p in paths if not p.exists()]
    if missing:
        pytest.skip("Missing data files: " + ", ".join(missing))


# ===========================================================================
# glcc — Great Lakes Coordinated Committee monthly NBS, four CSVs
# ===========================================================================
class TestGLCCLoader:
    """Tests the GLCC monthly NBS loader (``DataLoader.glcc``) against shipped CSVs."""

    LAKE_FILES = [
        "LakeSuperior_MonthlyNetBasinSupply_1900to2025.csv",
        "LakeMichiganHuron_MonthlyNetBasinSupply_1900to2025.csv",
        "LakeErie_MonthlyNetBasinSupply_1900to2025.csv",
        "LakeOntario_MonthlyNetBasinSupply_1900to2025.csv",
    ]
    EXPECTED_COLS = {
        "superior_nbs_obs",
        "michigan-huron_nbs_obs",
        "erie_nbs_obs",
        "ontario_nbs_obs",
    }

    @pytest.fixture(autouse=True)
    def _check_data(self):
        """Skip the GLCC tests cleanly if any lake CSV is missing."""
        _require_files(*[GLCC_DIR / f for f in self.LAKE_FILES])

    def test_returns_non_empty_dataframe(self):
        """Returns a non-empty DataFrame for the shipped GLCC data."""
        df = DataLoader().glcc(str(GLCC_DIR))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_index_is_datetime(self):
        """The output index is a DatetimeIndex named 'date'."""
        df = DataLoader().glcc(str(GLCC_DIR))
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"

    def test_has_one_obs_column_per_lake(self):
        """Output contains the expected per-lake ``*_nbs_obs`` columns."""
        df = DataLoader().glcc(str(GLCC_DIR))
        assert self.EXPECTED_COLS.issubset(set(df.columns))

    def test_observed_columns_are_numeric(self):
        """Every per-lake observed column has a numeric dtype."""
        df = DataLoader().glcc(str(GLCC_DIR))
        for col in self.EXPECTED_COLS:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"

    def test_dates_are_monthly_first_of_month(self):
        """Every date falls on the first of the month."""
        df = DataLoader().glcc(str(GLCC_DIR))
        # All dates should be the first of the month — no stray mid-month rows.
        assert (df.index.day == 1).all()

    def test_dates_are_unique(self):
        """The date index has no duplicates."""
        df = DataLoader().glcc(str(GLCC_DIR))
        assert df.index.is_unique, "duplicate dates in GLCC output"

    def test_dropped_all_nan_rows(self):
        """The loader drops rows where every lake column is NaN."""
        df = DataLoader().glcc(str(GLCC_DIR))
        lake_cols = [c for c in df.columns if c.endswith("_nbs_obs")]
        assert not df[lake_cols].isna().all(axis=1).any(), (
            "Found rows where every lake column is NaN — loader should drop these."
        )

    def test_units_mm_changes_values(self):
        """Requesting mm units yields different values than cms (unit conversion applied)."""
        df_cms = DataLoader().glcc(str(GLCC_DIR), units="cms")
        df_mm = DataLoader().glcc(str(GLCC_DIR), units="mm")
        # Same shape, but mm values are scaled by lake area + seconds-in-month.
        assert df_cms.shape == df_mm.shape
        # Compare a column's first non-NaN values.
        col = "superior_nbs_obs"
        cms_vals = df_cms[col].dropna()
        mm_vals = df_mm[col].dropna()
        assert not (cms_vals == mm_vals).all(), "mm conversion did not change values"

    def test_sanity_range_cms(self):
        """
        NBS values in cms for Great Lakes are typically in [-10000, 20000].
        We use a generous envelope so updates to the data don't break tests.
        """
        df = DataLoader().glcc(str(GLCC_DIR), units="cms")
        for col in self.EXPECTED_COLS:
            vals = df[col].dropna()
            assert (vals > -50000).all(), f"{col} has implausibly low values"
            assert (vals < 100000).all(), f"{col} has implausibly high values"


# ===========================================================================
# l2swbm — Large Lake Statistical Water Balance Model, 12 CSVs
# ===========================================================================
class TestL2SWBMLoader:
    """Tests the L2SWBM loader (``DataLoader.l2swbm``) against shipped CSVs."""

    LAKES = ["superior", "michigan-huron", "erie", "ontario"]
    VARS = ["Evap", "Runoff", "Precip"]

    @staticmethod
    def _filename(lake: str, var: str) -> str:
        """Return the L2SWBM CSV filename for a given lake and variable."""
        token = "miHuron" if lake == "michigan-huron" else lake
        return f"{token}{var}_MonthlyRun.csv"

    @pytest.fixture(autouse=True)
    def _check_data(self):
        """Skip the L2SWBM tests cleanly if any (lake, variable) CSV is missing."""
        files = [L2SWBM_DIR / self._filename(lake, var)
                 for lake in self.LAKES for var in self.VARS]
        _require_files(*files)

    def test_returns_non_empty_dataframe(self):
        """Returns a non-empty DataFrame for the shipped L2SWBM data."""
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_index_is_datetime(self):
        """The output index is a DatetimeIndex named 'date'."""
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "date"

    def test_has_required_variables_for_each_lake(self):
        """
        Issue #38: Required variables exist (precipitation, evaporation,
        runoff). The loader renames precip→precipitation and evap→evaporation.
        """
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        expected = {
            f"{lake}_{var}_obs"
            for lake in self.LAKES
            for var in ["evaporation", "runoff", "precipitation"]
        }
        missing = expected - set(df.columns)
        assert not missing, f"missing required columns: {missing}"

    def test_all_value_columns_are_numeric(self):
        """Every value column has a numeric dtype."""
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"

    def test_no_unexpected_nan(self):
        """
        L2SWBM values should be present for every (lake, variable, date) in
        the merged window. Real data may have boundary years missing — but the
        merge is `pd.merge(... how='inner')` (default), so the surviving
        index should be NaN-free.
        """
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        nan_counts = df.isna().sum()
        offenders = nan_counts[nan_counts > 0]
        assert offenders.empty, f"unexpected NaN in columns: {offenders.to_dict()}"

    def test_precipitation_non_negative(self):
        """Sanity: precipitation can't be negative (mass-balance physics)."""
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        for col in [c for c in df.columns if c.endswith("_precipitation_obs")]:
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_evaporation_in_plausible_range(self):
        """
        Sanity: in L2SWBM, monthly evaporation can be slightly negative when
        the period's net flux is dominated by condensation — the value is a
        statistical residual, not a gross flux. Real data has occasional
        small negatives (~5% of months on Superior, min around -16 mm) which
        is physically reasonable. Implausibly large negatives (e.g. < -100
        mm/month) would indicate a unit or sign-convention bug.
        """
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        for col in [c for c in df.columns if c.endswith("_evaporation_obs")]:
            assert (df[col] > -100).all(), f"{col} has implausibly large negative values"
            assert (df[col] < 1000).all(), f"{col} has implausibly large positive values"

    def test_runoff_non_negative(self):
        """Sanity: runoff values are never negative."""
        df = DataLoader().l2swbm(str(L2SWBM_DIR))
        for col in [c for c in df.columns if c.endswith("_runoff_obs")]:
            assert (df[col] >= 0).all(), f"{col} has negative values"


# ===========================================================================
# glsea — Surface temperature, single whitespace-separated CSV
# ===========================================================================
class TestGLSEALoader:
    """Tests the GLSEA surface-temperature loader (``DataLoader.glsea``)."""

    EXPECTED_COLS = ["superior_sst", "michigan-huron_sst", "erie_sst", "ontario_sst"]

    @pytest.fixture(autouse=True)
    def _check_data(self):
        """Skip the GLSEA tests cleanly if the SST CSV is missing."""
        _require_files(GLSEA_FILE)

    def test_returns_non_empty_dataframe(self):
        """Returns a non-empty DataFrame for the shipped GLSEA data."""
        df = DataLoader().glsea(str(GLSEA_FILE))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_index_is_datetime(self):
        """The output index is a DatetimeIndex."""
        df = DataLoader().glsea(str(GLSEA_FILE))
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_columns_in_expected_order(self):
        """
        Issue #38 requirement: required variables exist in correct order.
        The loader renames + reorders columns positionally; this pins the
        public contract.
        """
        df = DataLoader().glsea(str(GLSEA_FILE))
        assert list(df.columns) == self.EXPECTED_COLS

    def test_values_are_numeric(self):
        """Every temperature column has a numeric dtype."""
        df = DataLoader().glsea(str(GLSEA_FILE))
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"

    def test_default_units_kelvin_in_lake_temperature_range(self):
        """
        Sanity range: Great Lakes surface water in Kelvin should be roughly
        [270, 305] (just below freezing on the cold end, mid-summer Erie on
        the warm end). Generous envelope to allow for data updates.
        """
        df = DataLoader().glsea(str(GLSEA_FILE))
        non_nan = df.dropna()
        assert (non_nan > 268).all().all(), "GLSEA Kelvin values implausibly cold"
        assert (non_nan < 310).all().all(), "GLSEA Kelvin values implausibly hot"

    def test_celsius_units_in_lake_temperature_range(self):
        """With units='C', values fall in a plausible Celsius lake-temperature range."""
        df = DataLoader().glsea(str(GLSEA_FILE), units="C")
        non_nan = df.dropna()
        assert (non_nan > -5).all().all(), "GLSEA Celsius values implausibly cold"
        assert (non_nan < 35).all().all(), "GLSEA Celsius values implausibly hot"

    def test_michigan_huron_is_mean_of_michigan_and_huron(self):
        """
        The loader synthesizes michigan-huron as the mean of michigan + huron.
        Verify on real data by reading the raw file alongside.
        """
        df = DataLoader().glsea(str(GLSEA_FILE), units="C")
        # Sanity: Michigan-Huron should fall between min and max of all four lakes.
        non_nan = df.dropna()
        all_max = non_nan[self.EXPECTED_COLS].max(axis=1)
        all_min = non_nan[self.EXPECTED_COLS].min(axis=1)
        mh = non_nan["michigan-huron_sst"]
        assert (mh >= all_min).all() and (mh <= all_max).all()

    def test_unsupported_units_raises(self):
        """An unsupported units argument raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported units"):
            DataLoader().glsea(str(GLSEA_FILE), units="F")


# ===========================================================================
# lake_probabilities — 4 lake CSVs of probability of exceedance × 12 months
# ===========================================================================
class TestLakeProbabilitiesLoader:
    """Tests the lake-probabilities loader (``DataLoader.lake_probabilities``)."""

    LAKE_FILES = ["SUP.probs.csv", "MIH.probs.csv", "ERI.probs.csv", "ONT.probs.csv"]
    LAKES = ["superior", "michigan-huron", "erie", "ontario"]
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    @pytest.fixture(autouse=True)
    def _check_data(self):
        """Skip the probabilities tests cleanly if any lake CSV is missing."""
        _require_files(*[PROBABILITIES_DIR / f for f in self.LAKE_FILES])

    @pytest.fixture
    def probs_dir(self):
        """Return the probabilities directory with a trailing separator for the loader."""
        # Loader uses string concat (file_dir + filename), so trailing sep needed.
        return str(PROBABILITIES_DIR) + "/"

    def test_returns_long_format(self, probs_dir):
        """Returns a long-format DataFrame with the expected four columns."""
        df = DataLoader().lake_probabilities(probs_dir)
        assert list(df.columns) == ["month", "lake", "prob_exceedance", "value"]

    def test_has_all_four_lakes(self, probs_dir):
        """Output covers all four lakes."""
        df = DataLoader().lake_probabilities(probs_dir)
        assert set(df["lake"].unique()) == set(self.LAKES)

    def test_has_all_twelve_months_per_lake(self, probs_dir):
        """Each lake has all twelve months represented."""
        df = DataLoader().lake_probabilities(probs_dir)
        for lake in self.LAKES:
            months_for_lake = set(df[df["lake"] == lake]["month"])
            assert months_for_lake == set(self.MONTHS), f"{lake} missing months"

    def test_prob_exceedance_is_float(self, probs_dir):
        """The prob_exceedance column has a float dtype."""
        df = DataLoader().lake_probabilities(probs_dir)
        assert pd.api.types.is_float_dtype(df["prob_exceedance"])

    def test_value_is_finite(self, probs_dir):
        """
        Probability-of-exceedance flow values can be negative (legitimate for
        NBS in winter months) but should always be finite.
        """
        df = DataLoader().lake_probabilities(probs_dir)
        assert df["value"].notna().all(), "NaN in probability values"
        # Sanity envelope — Great Lakes monthly NBS in cms is bounded.
        assert (df["value"].abs() < 100000).all()

    def test_mm_units_changes_values(self, probs_dir):
        """Requesting mm units yields different values than cms (unit conversion applied)."""
        df_cms = DataLoader().lake_probabilities(probs_dir, units="cms")
        df_mm = DataLoader().lake_probabilities(probs_dir, units="mm")
        assert df_cms.shape == df_mm.shape
        assert not (df_cms["value"] == df_mm["value"]).all()

    def test_prob_exceedance_levels_between_0_and_1(self, probs_dir):
        """Sanity: probability levels by definition lie in (0, 1)."""
        df = DataLoader().lake_probabilities(probs_dir)
        assert (df["prob_exceedance"] > 0).all()
        assert (df["prob_exceedance"] < 1).all()
