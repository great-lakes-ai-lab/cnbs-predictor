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
    def test_returns_true_on_200(self):
        fake_response = MagicMock(status_code=200)
        with patch("src.utilities.requests.head", return_value=fake_response):
            assert check_url_exists("https://example.com") is True

    def test_returns_false_on_non_200(self):
        fake_response = MagicMock(status_code=404)
        with patch("src.utilities.requests.head", return_value=fake_response):
            assert check_url_exists("https://example.com/missing") is False

    def test_returns_false_on_request_exception(self):
        with patch(
            "src.utilities.requests.head",
            side_effect=requests.ConnectionError("boom"),
        ):
            assert check_url_exists("https://example.com") is False


# ---------------------------------------------------------------------------
# get_first_forecast_month
# ---------------------------------------------------------------------------
class TestGetFirstForecastMonth:
    def test_before_26th_returns_current_month(self):
        result = get_first_forecast_month(today=datetime(2024, 6, 10))
        assert result == "06-2024"

    def test_on_25th_still_current_month(self):
        # Boundary: rule says "before 26th" -> current month
        result = get_first_forecast_month(today=datetime(2024, 6, 25))
        assert result == "06-2024"

    def test_on_26th_rolls_to_next_month(self):
        result = get_first_forecast_month(today=datetime(2024, 6, 26))
        assert result == "07-2024"

    def test_after_26th_rolls_to_next_month(self):
        result = get_first_forecast_month(today=datetime(2024, 6, 30))
        assert result == "07-2024"

    def test_december_rollover_to_next_year(self):
        result = get_first_forecast_month(today=datetime(2024, 12, 27))
        assert result == "01-2025"

    def test_default_today_runs_without_error(self):
        # No assertion on value (date-dependent), just that the default path works.
        result = get_first_forecast_month()
        assert isinstance(result, str)
        assert len(result) == 7 and result[2] == "-"


# ---------------------------------------------------------------------------
# get_date_range
# ---------------------------------------------------------------------------
class TestGetDateRange:
    def test_manual_valid_range_returns_datetimes_and_index(self):
        start, end, idx = get_date_range(
            auto="no", start_date="01-01-2024", end_date="01-05-2024"
        )
        assert start == datetime(2024, 1, 1)
        assert end == datetime(2024, 1, 5)
        assert len(idx) == 5  # daily, inclusive

    def test_manual_same_start_and_end_does_not_raise(self):
        # The function prints "up-to-date" but should not raise.
        start, end, idx = get_date_range(
            auto="no", start_date="01-01-2024", end_date="01-01-2024"
        )
        assert start == end
        assert len(idx) == 1

    def test_manual_end_before_start_raises(self):
        with pytest.raises(ValueError, match="End date cannot be older"):
            get_date_range(auto="no", start_date="01-05-2024", end_date="01-01-2024")

    def test_auto_yes_without_db_raises(self):
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
    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "new_subdir"
        assert not target.exists()
        create_directory(str(target))
        assert target.is_dir()

    def test_no_error_when_directory_exists(self, tmp_path):
        target = tmp_path / "existing"
        target.mkdir()
        # Should print and return without raising
        create_directory(str(target))
        assert target.is_dir()

    def test_creates_nested_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        create_directory(str(target))
        assert target.is_dir()


# ---------------------------------------------------------------------------
# get_files
# ---------------------------------------------------------------------------
class TestGetFiles:
    def _populate(self, tmp_path):
        (tmp_path / "alpha.csv").write_text("x")
        (tmp_path / "beta.csv").write_text("x")
        (tmp_path / "alpha.txt").write_text("x")
        (tmp_path / "gamma.log").write_text("x")
        return tmp_path

    def test_suffix_match_returns_csv_files(self, tmp_path):
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="suffix", identifier=".csv")
        expected = [
            str(d / "alpha.csv"),
            str(d / "beta.csv"),
        ]
        assert sorted(result) == sorted(expected)

    def test_prefix_match_returns_alpha_files(self, tmp_path):
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="prefix", identifier="alpha")
        expected = [
            str(d / "alpha.csv"),
            str(d / "alpha.txt"),
        ]
        assert sorted(result) == sorted(expected)

    def test_no_match_returns_empty_list(self, tmp_path):
        d = self._populate(tmp_path)
        result = get_files(str(d), affix="suffix", identifier=".nope")
        assert result == []
