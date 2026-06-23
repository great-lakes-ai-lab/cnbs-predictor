# tests/unit/test_utilities.py
"""
Unit tests for src/utilities.py.

These are pure-logic tests — network calls in ``check_url_exists`` are mocked,
file-system functions use pytest's ``tmp_path``. No external services are
required to run this file.
"""

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.utilities import (
    check_url_exists,
    get_first_forecast_month,
    get_date_range,
    create_directory,
    get_files,
)


# ---------------------------------------------------------------------------
# check_url_exists  (mocked — does not hit the network)
# ---------------------------------------------------------------------------
class TestCheckUrlExists:
    """Tests for ``check_url_exists`` with the HTTP HEAD call mocked."""

    def test_returns_true_on_200(self):
        """Returns True when the HEAD request responds with status 200."""
        fake_response = MagicMock(status_code=200)
        with patch("src.utilities.requests.head", return_value=fake_response):
            assert check_url_exists("https://example.com") is True

    def test_returns_false_on_non_200(self):
        """Returns False when the HEAD request responds with a non-200 status."""
        fake_response = MagicMock(status_code=404)
        with patch("src.utilities.requests.head", return_value=fake_response):
            assert check_url_exists("https://example.com/missing") is False

    def test_returns_false_on_request_exception(self):
        """Returns False when the HEAD request raises a RequestException."""
        with patch(
            "src.utilities.requests.head",
            side_effect=requests.ConnectionError("boom"),
        ):
            assert check_url_exists("https://example.com") is False


# ---------------------------------------------------------------------------
# get_first_forecast_month
# ---------------------------------------------------------------------------
class TestGetFirstForecastMonth:
    """Tests the 26th-of-month cutoff rule in ``get_first_forecast_month``."""

    def test_before_26th_returns_current_month(self):
        """Before the 26th, returns the current month."""
        result = get_first_forecast_month(today=datetime(2024, 6, 10))
        assert result == "06-2024"

    def test_on_25th_still_current_month(self):
        """On the 25th (boundary), still returns the current month."""
        # Boundary: rule says "before 26th" -> current month
        result = get_first_forecast_month(today=datetime(2024, 6, 25))
        assert result == "06-2024"

    def test_on_26th_rolls_to_next_month(self):
        """On the 26th (boundary), rolls over to the next month."""
        result = get_first_forecast_month(today=datetime(2024, 6, 26))
        assert result == "07-2024"

    def test_after_26th_rolls_to_next_month(self):
        """After the 26th, rolls over to the next month."""
        result = get_first_forecast_month(today=datetime(2024, 6, 30))
        assert result == "07-2024"

    def test_december_rollover_to_next_year(self):
        """A late-December date rolls over to January of the next year."""
        result = get_first_forecast_month(today=datetime(2024, 12, 27))
        assert result == "01-2025"

    def test_default_today_runs_without_error(self):
        """Calling with the default (real) today returns a well-formed MM-YYYY string."""
        # No assertion on value (date-dependent), just that the default path works.
        result = get_first_forecast_month()
        assert isinstance(result, str)
        assert len(result) == 7 and result[2] == "-"


# ---------------------------------------------------------------------------
# get_date_range
# ---------------------------------------------------------------------------
class TestGetDateRange:
    """Tests manual and auto date-range resolution in ``get_date_range``."""

    def test_manual_valid_range_returns_datetimes_and_index(self):
        """A valid manual range returns parsed start/end datetimes and an inclusive daily index."""
        start, end, idx = get_date_range(
            auto="no", start_date="01-01-2024", end_date="01-05-2024"
        )
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 1, 5)
        assert len(idx) == 5  # daily, inclusive

    def test_manual_same_start_and_end_does_not_raise(self):
        """Equal start and end dates yield a single-day index without raising."""
        # The function prints "up-to-date" but should not raise.
        start, end, idx = get_date_range(
            auto="no", start_date="01-01-2024", end_date="01-01-2024"
        )
        assert start == end
        assert len(idx) == 1

    def test_manual_end_before_start_raises(self):
        """An end date earlier than the start date raises ValueError."""
        with pytest.raises(ValueError, match="End date cannot be older"):
            get_date_range(auto="no", start_date="01-05-2024", end_date="01-01-2024")

    def test_auto_yes_without_db_raises(self):
        """auto='yes' without a database object raises ValueError."""
        with pytest.raises(ValueError, match="Database object must be provided"):
            get_date_range(auto="yes", db=None)

    def test_auto_yes_happy_path_currently_broken(self):
        """
        Known bug: ``get_date_range`` references ``timedelta`` in the auto='yes'
        branch but never imports it. The call should raise ``NameError`` until
        the import is fixed. This test pins the current behavior so the bug is
        visible; flip the assertion to a happy-path check once the import is added.
        """
        fake_db = MagicMock()
        fake_db.get_next_run.return_value = datetime(2024, 1, 1)

        with pytest.raises(NameError, match="timedelta"):
            get_date_range(auto="yes", db=fake_db)


# ---------------------------------------------------------------------------
# create_directory
# ---------------------------------------------------------------------------
class TestCreateDirectory:
    """Tests directory creation behavior of ``create_directory``."""

    def test_creates_missing_directory(self, tmp_path):
        """Creates a directory that does not yet exist."""
        target = tmp_path / "new_subdir"
        assert not target.exists()
        create_directory(str(target))
        assert target.is_dir()

    def test_no_error_when_directory_exists(self, tmp_path):
        """Does not raise when the target directory already exists."""
        target = tmp_path / "existing"
        target.mkdir()
        # Should print and return without raising
        create_directory(str(target))
        assert target.is_dir()

    def test_creates_nested_directory(self, tmp_path):
        """Creates intermediate parent directories for a nested path."""
        target = tmp_path / "a" / "b" / "c"
        create_directory(str(target))
        assert target.is_dir()


# ---------------------------------------------------------------------------
# get_files
# ---------------------------------------------------------------------------
class TestGetFiles:
    """Tests suffix/prefix filtering of directory listings by ``get_files``."""

    def _populate(self, tmp_path):
        """Create a fixed set of mixed-extension files in ``tmp_path``."""
        (tmp_path / "alpha.csv").write_text("x")
        (tmp_path / "beta.csv").write_text("x")
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "gamma.log").write_text("x")
        return tmp_path

    def test_suffix_match_returns_csv_files(self, tmp_path):
        """Suffix match on '.csv' returns only the .csv files."""
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="suffix", identifier=".csv")
        expected = [
            str(d / "alpha.csv"),
            str(d / "beta.csv"),
        ]
        assert sorted(result) == sorted(expected)

    def test_prefix_match_returns_alpha_files(self, tmp_path):
        """Prefix match on 'alpha' returns only files whose name starts with alpha."""
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="prefix", identifier="alpha")
        expected = [
            str(d / "alpha.csv"),
            str(d / "alpha.txt"),
        ]
        assert sorted(result) == sorted(expected)

    def test_no_match_returns_empty_list(self, tmp_path):
        """An identifier matching no files returns an empty list."""
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="suffix", identifier=".nope")
        assert result == []
