# tests/unit/test_database_utils.py
"""
Unit tests for ``CFSDatabase`` loading and indexing in ``src/database_utils.py``.

Strategy: real SQLite databases written to pytest's ``tmp_path``. SQLite is a
local file, so these are fast and deterministic — no network, no fixtures, no
mocking of the database layer.

The focus is ``load(start_date=...)``, which pushes the forecast-month filter
into SQL so notebook 3 does not transfer an entire table across a network
share. The contract that makes that safe is *equivalence*: the SQL-filtered
result must match what loading everything and filtering in pandas produces.
Those equivalence tests are the important ones here.
"""

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from src.data_processor import ForecastTransformer
from src.database_utils import CFSDatabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _year_month_frame():
    """
    Build a frame in the ``year``/``month`` layout notebook 2 writes and
    notebook 3 reads, spanning 2026-05 through 2027-02 so a mid-range
    threshold has rows on both sides.
    """
    months = [
        (2026, 5), (2026, 6), (2026, 7), (2026, 8), (2026, 12),
        (2027, 1), (2027, 2),
    ]
    return pd.DataFrame({
        "cfs_run": ["2026050100"] * len(months),
        "month": [m for _, m in months],
        "year": [y for y, _ in months],
        "model": ["RF"] * len(months),
        "lake": ["erie"] * len(months),
        "component": ["nbs"] * len(months),
        "value [mm]": [float(i) for i in range(len(months))],
    })


def _ensemble_frame():
    """
    Build an ensemble-shaped frame: many rows sharing each ``(year, month)``.

    The real database holds an ensemble — every forecast month has one row per
    (cfs_run, model, lake, component) combination, so ``(year, month)`` is far
    from unique. Fixtures with one row per month cannot catch a filter that
    collapses duplicates, hence this one: 2 runs x 3 models x 2 lakes x 2
    components = 24 rows per month.
    """
    runs = ["2026010100", "2026010106"]
    models = ["RF", "GP", "XGB"]
    lakes = ["superior", "erie"]
    components = ["precipitation", "nbs"]
    months = [(2026, 6), (2026, 7), (2027, 1)]

    rows = [
        (run, month, year, model, lake, component, 1.0)
        for year, month in months
        for run in runs
        for model in models
        for lake in lakes
        for component in components
    ]
    return pd.DataFrame(rows, columns=[
        "cfs_run", "month", "year", "model", "lake", "component", "value [mm]",
    ])


def _forecast_month_frame():
    """Build a frame in the ``forecast_month`` ('YYYY-MM') layout ``pivot`` produces."""
    fmonths = ["2026-05", "2026-06", "2026-07", "2026-10", "2027-01"]
    return pd.DataFrame({
        "cfs_run": ["2026050100"] * len(fmonths),
        "forecast_month": fmonths,
        "model": ["RF"] * len(fmonths),
        "lake": ["erie"] * len(fmonths),
        "precipitation": [float(i) for i in range(len(fmonths))],
    })


def _write(tmp_path, df, table="cnbs_forecast", name="test.db"):
    """Write ``df`` to a fresh SQLite database and return its CFSDatabase."""
    db = CFSDatabase(str(tmp_path / name), table)
    db.add_df(df)
    return db


def _index_names(database):
    """Return the user-defined index names present in the database file."""
    conn = sqlite3.connect(database)
    try:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        conn.close()


# ===========================================================================
# load() — backward compatibility
# ===========================================================================
class TestLoadUnfiltered:
    """``load()`` with no arguments keeps its original whole-table behavior."""

    def test_returns_every_row(self, tmp_path):
        """No start_date means no WHERE clause: all rows come back."""
        df = _year_month_frame()
        out = _write(tmp_path, df).load()
        assert len(out) == len(df)

    def test_preserves_column_order(self, tmp_path):
        """SELECT * keeps the written column order."""
        df = _year_month_frame()
        out = _write(tmp_path, df).load()
        assert list(out.columns) == list(df.columns)

    def test_works_on_table_without_date_columns(self, tmp_path):
        """
        A table with neither layout is still fully loadable — the date columns
        are only required when start_date is supplied.
        """
        df = pd.DataFrame({"cfs_run": ["2026050100"], "value": [1.0]})
        out = _write(tmp_path, df, table="odd_table").load()
        assert len(out) == 1

class TestLoadFiltered:
    """load() with filter_by and start_date applies date filtering."""

    def test_filters_by_year_month(self, tmp_path):
        """Filters using separate year and month columns."""
        df = pd.DataFrame({
            "year": [2025, 2025, 2026, 2026],
            "month": [11, 12, 1, 5],
            "value": [1, 2, 3, 4]
        })

        out = _write(tmp_path, df).load(
            filter_by=["year", "month"],
            start_date="2026-01"
        )

        assert len(out) == 2
        assert out["year"].tolist() == [2026, 2026]
        assert out["month"].tolist() == [1, 5]

    def test_filters_by_forecast_month(self, tmp_path):
        """Filters MM-YYYY forecast_month values chronologically."""
        df = pd.DataFrame({
            "forecast_month": [
                "2025-11",
                "2025-12",
                "2026-01",
                "2026-05"
            ],
            "value": [1, 2, 3, 4]
        })

        out = _write(tmp_path, df).load(
            filter_by="forecast_month",
            start_date="2026-01"
        )

        assert len(out) == 2
        assert out["forecast_month"].tolist() == [
            "2026-01", "2026-05"
        ]

    def test_filters_by_cfs_run(self, tmp_path):
        """Filters YYYYMMDDHH cfs_run values."""
        df = pd.DataFrame({
            "cfs_run": [
                "2025120100",
                "2026010100",
                "2026050100",
                "2026060100"
            ],
            "value": [1, 2, 3, 4]
        })

        out = _write(tmp_path, df).load(
            filter_by="cfs_run",
            start_date="2026-05"
        )

        assert len(out) == 2
        assert out["cfs_run"].tolist() == [
            "2026050100", "2026060100"
        ]

    def test_start_date_requires_filter_by(self, tmp_path):
        """start_date without filter_by raises an error."""
        database = _write(tmp_path, _year_month_frame())

        with pytest.raises(ValueError, match="filter_by"):
            database.load(start_date="2026-01")

    def test_invalid_filter_column_reports_available_columns(self, tmp_path):
        """Invalid columns produce a useful error."""
        database = _write(tmp_path, _year_month_frame())

        with pytest.raises(ValueError, match="not found in table"):
            database.load(
                filter_by="invalid_column",
                start_date="2026-01"
            )

    def test_unsupported_filter_by_raises_error(self, tmp_path):
        """Existing but unsupported date columns raise an error."""
        df = pd.DataFrame({
            "value": [1, 2, 3]
        })

        database = _write(tmp_path, df)

        with pytest.raises(ValueError, match="Unsupported filter_by"):
            database.load(
                filter_by="value",
                start_date="2026-01"
            )

    def test_error_includes_first_column_value(self, tmp_path):
        """Malformed forecast_month values report the first value."""
        df = pd.DataFrame({
            "forecast_month": ["01/2026", "02/2026"],
            "value": [1, 2]
        })

        database = _write(tmp_path, df)

        with pytest.raises(ValueError) as exc:
            database.load(
                filter_by="forecast_month",
                start_date="2026-01"
            )

        assert "01/2026" in str(exc.value)
        assert "forecast_month" in str(exc.value)
        assert "YYYY-MM" in str(exc.value)

    def test_error_includes_first_cfs_run_value(self, tmp_path):
        """Malformed cfs_run values report the first value."""
        df = pd.DataFrame({
            "cfs_run": ["2026-05-01", "2026-06-01"],
            "value": [1, 2]
        })

        database = _write(tmp_path, df)

        with pytest.raises(ValueError) as exc:
            database.load(
                filter_by="cfs_run",
                start_date="2026-05"
            )

        assert "2026-05-01" in str(exc.value)
        assert "cfs_run" in str(exc.value)
        assert "YYYYMMDDHH" in str(exc.value)

    def test_error_includes_first_year_month_values(self, tmp_path):
        """Invalid year/month values report the first available values."""
        df = pd.DataFrame({
            "year": [2026, 2026],
            "month": [13, 14],
            "value": [1, 2]
        })

        database = _write(tmp_path, df)

        with pytest.raises(ValueError) as exc:
            database.load(
                filter_by=["year", "month"],
                start_date="2026-01"
            )

        assert "year" in str(exc.value)
        assert "month" in str(exc.value)
        assert "13" in str(exc.value)

    def test_invalid_start_date_raises_error(self, tmp_path):
        """Invalid start_date formats raise a useful error."""
        df = pd.DataFrame({
            "forecast_month": ["2026-01", "2026-02"],
            "value": [1, 2]
        })

        database = _write(tmp_path, df)

        with pytest.raises(ValueError, match="Invalid start_date"):
            database.load(
                filter_by="forecast_month",
                start_date="not-a-date"
            )

    def test_filters_exclude_rows_before_start_date(self, tmp_path):
        """The starting month itself is included."""
        df = pd.DataFrame({
            "forecast_month": [
                "2025-12",
                "2026-01",
                "2026-02",
                "2026-03"
            ],
            "value": [1, 2, 3, 4]
        })

        out = _write(tmp_path, df).load(
            filter_by="forecast_month",
            start_date="2026-02"
        )

        assert out["forecast_month"].tolist() == [
            "2026-02", "2026-03"
        ]

    def test_empty_table_returns_empty_dataframe(self, tmp_path):
        """An empty table can still be filtered without errors."""
        df = pd.DataFrame({
            "forecast_month": pd.Series(dtype=str),
            "value": pd.Series(dtype=float)
        })

        out = _write(tmp_path, df).load(
            filter_by="forecast_month",
            start_date="2026-01"
        )

        assert out.empty
        assert list(out.columns) == ["forecast_month", "value"]

# ===========================================================================
# Equivalence: SQL pushdown must match filtering in pandas
# ===========================================================================


class TestPushdownMatchesPandasFilter:
    """
    ``load(start_date=X, filter_by=Y)`` must return exactly what
    ``ForecastTransformer(load()).filter(X)`` returns.

    The database can contain either:
      - cfs_run, year, month
      - cfs_run, forecast_month
    """

    @staticmethod
    def _assert_same(pushed, in_pandas, key):
        """
        Compare the two paths on rows, columns and values.

        ``check_dtype=False`` because an empty SQL result comes back with
        ``object`` columns, while filtering a populated frame down to nothing
        in pandas preserves the original dtypes.
        """
        pd.testing.assert_frame_equal(
            pushed.sort_values(key).reset_index(drop=True),
            in_pandas.sort_values(key).reset_index(drop=True),
            check_dtype=False,
        )

    @pytest.mark.parametrize(
        "start_date",
        ["05-2026", "07-2026", "01-2027", "01-2030"],
    )
    def test_year_month_layout_matches(self, tmp_path, start_date):
        """Equivalence for a table containing cfs_run, year, and month."""
        db = _write(tmp_path, _year_month_frame())

        self._assert_same(
            db.load(
                start_date=start_date,
                filter_by=["year", "month"],
            ),
            ForecastTransformer(db.load()).filter(start_date),
            ["cfs_run", "year", "month"],
        )

    @pytest.mark.parametrize(
        "start_date",
        ["05-2026", "07-2026", "01-2027"],
    )
    def test_forecast_month_layout_matches(self, tmp_path, start_date):
        """Equivalence for a table containing cfs_run and forecast_month."""
        db = _write(tmp_path, _forecast_month_frame())

        self._assert_same(
            db.load(
                start_date=start_date,
                filter_by="forecast_month",
            ),
            ForecastTransformer(db.load()).filter(start_date),
            ["cfs_run", "forecast_month"],
        )

    def test_ensemble_members_all_survive(self, tmp_path):
        """
        Every ensemble member of a kept month is returned.

        Filtering selects whole forecast months; it must not collapse rows
        sharing the same year and month.
        """
        db = _write(tmp_path, _ensemble_frame())
        out = db.load(
            start_date="07-2026",
            filter_by=["year", "month"],
        )

        per_month = out.groupby(["year", "month"]).size().to_dict()
        assert per_month == {
            (2026, 7): 24,
            (2027, 1): 24,
        }

    def test_ensemble_layout_matches_pandas_filter(self, tmp_path):
        """Equivalence holds on ensemble-shaped data, duplicates included."""
        db = _write(tmp_path, _ensemble_frame())

        self._assert_same(
            db.load(
                start_date="07-2026",
                filter_by=["year", "month"],
            ),
            ForecastTransformer(db.load()).filter("07-2026"),
            [
                "cfs_run",
                "year",
                "month",
                "model",
                "lake",
                "component",
            ],
        )

    def test_dtypes_match_when_rows_survive(self, tmp_path):
        """For non-empty results, both paths should agree on dtypes."""
        db = _write(tmp_path, _year_month_frame())

        pushed = db.load(
            start_date="07-2026",
            filter_by=["year", "month"],
        )
        in_pandas = ForecastTransformer(db.load()).filter("07-2026")

        assert not pushed.empty
        assert list(pushed.dtypes) == list(in_pandas.dtypes)

# ===========================================================================
# create_indexes()
# ===========================================================================
class TestCreateIndexes:
    """
    The index is what makes the WHERE clause reduce I/O — without it SQLite
    scans every page regardless. ``add_df`` creates it so readers of a shared
    database do not have to.
    """

    def test_add_df_indexes_year_month_layout(self, tmp_path):
        """Writing the year/month layout leaves a (year, month) index behind."""
        db = _write(tmp_path, _year_month_frame())
        assert "idx_cnbs_forecast_year_month" in _index_names(db.database)

    def test_add_df_indexes_forecast_month_layout(self, tmp_path):
        """Writing the forecast_month layout leaves a forecast_month index behind."""
        db = _write(tmp_path, _forecast_month_frame(), table="pivoted")
        assert "idx_pivoted_forecast_month" in _index_names(db.database)

    def test_no_index_when_table_has_no_date_columns(self, tmp_path):
        """Tables without date columns get no index rather than an error."""
        df = pd.DataFrame({"cfs_run": ["2026050100"], "value": [1.0]})
        db = _write(tmp_path, df, table="odd_table")
        assert _index_names(db.database) == set()

    def test_is_idempotent(self, tmp_path):
        """Repeated calls are safe and converge on the same index set."""
        db = _write(tmp_path, _year_month_frame())
        first = db.create_indexes()
        second = db.create_indexes()
        assert first == second
        assert _index_names(db.database) == set(first)

    def test_index_survives_a_replace_write(self, tmp_path):
        """
        ``if_exists='replace'`` drops the table *and* its indexes. Because
        add_df re-indexes after writing, the index is restored.
        """
        db = _write(tmp_path, _year_month_frame())
        db.add_df(_year_month_frame(), if_exists="replace")
        assert "idx_cnbs_forecast_year_month" in _index_names(db.database)

    def test_index_failure_does_not_discard_the_write(self, tmp_path, capsys):
        """
        Indexing is an optimization, so a failure must not surface as an
        exception from ``add_df`` — the caller would retry and append the rows
        twice. The data stays written and a warning is printed instead.
        """
        df = _year_month_frame()
        db = CFSDatabase(str(tmp_path / "readonly.db"), "cnbs_forecast")

        with patch.object(
            CFSDatabase, "create_indexes",
            side_effect=sqlite3.DatabaseError("attempt to write a readonly database"),
        ):
            db.add_df(df)  # must not raise

        assert len(db.load()) == len(df)
        assert "WARNING" in capsys.readouterr().out

    def test_query_plan_uses_the_index(self, tmp_path):
        """
        The point of the index is that a selective filtered query *searches*
        rather than scans — a plan that says SCAN means the predicate stopped
        being index-friendly and the network win is gone.

        Uses a table spanning many months and a late threshold, so selecting
        via the index is unambiguously the cheaper plan. (On a tiny table, or
        one where the filter matches most rows, SQLite correctly prefers a
        scan; that choice is not what this test is about.)
        """
        months = [(2026, m) for m in range(1, 13)] + [(2027, m) for m in range(1, 13)]
        rows = months * 40  # ~960 rows, evenly spread across 24 months
        df = pd.DataFrame({
            "cfs_run": ["2026010100"] * len(rows),
            "month": [m for _, m in rows],
            "year": [y for y, _ in rows],
            "value [mm]": [1.0] * len(rows),
        })
        db = _write(tmp_path, df)

        conn = sqlite3.connect(db.database)
        try:
            # ANALYZE so the decision reflects real statistics rather than
            # the planner's no-stats defaults.
            conn.execute("ANALYZE")
            plan = "\n".join(
                str(row)
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM cnbs_forecast "
                    "WHERE year > 2027 OR (year = 2027 AND month >= 12)"
                )
            )
        finally:
            conn.close()

        assert "idx_cnbs_forecast_year_month" in plan, plan
