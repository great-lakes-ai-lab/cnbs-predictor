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


# ===========================================================================
# load(start_date=...) — SQL-side filtering
# ===========================================================================
class TestLoadFilteredYearMonth:
    """
    On the ``year``/``month`` layout, the filter becomes
    ``WHERE year > ? OR (year = ? AND month >= ?)``.
    """

    def test_drops_earlier_months_keeps_later(self, tmp_path):
        """Rows before the threshold month are excluded; the rest are kept."""
        out = _write(tmp_path, _year_month_frame()).load(start_date="07-2026")
        kept = set(zip(out["year"], out["month"]))
        assert kept == {(2026, 7), (2026, 8), (2026, 12), (2027, 1), (2027, 2)}

    def test_threshold_month_is_inclusive(self, tmp_path):
        """The start month itself is retained (>=, not >)."""
        out = _write(tmp_path, _year_month_frame()).load(start_date="07-2026")
        assert (2026, 7) in set(zip(out["year"], out["month"]))

    def test_year_rollover_not_compared_by_month_alone(self, tmp_path):
        """
        January 2027 (month=1) must survive a July 2026 threshold. A naive
        ``month >= 7`` predicate would wrongly drop it — this pins the
        (year, month) pair comparison.
        """
        out = _write(tmp_path, _year_month_frame()).load(start_date="07-2026")
        assert (2027, 1) in set(zip(out["year"], out["month"]))

    @pytest.mark.parametrize("start_date", ["07-2026", "2026-07", "2026-07-01"])
    def test_accepts_the_date_formats_in_use(self, tmp_path, start_date):
        """
        ``get_first_forecast_month()`` returns 'MM-YYYY'; 'YYYY-MM' and full
        dates are also accepted since parsing goes through pandas.Period.
        """
        out = _write(tmp_path, _year_month_frame()).load(start_date=start_date)
        assert len(out) == 5

    def test_threshold_after_all_data_returns_empty(self, tmp_path):
        """A threshold past every row yields an empty frame, not an error."""
        out = _write(tmp_path, _year_month_frame()).load(start_date="01-2030")
        assert out.empty

    def test_threshold_before_all_data_returns_everything(self, tmp_path):
        """A threshold before every row keeps the whole table."""
        df = _year_month_frame()
        out = _write(tmp_path, df).load(start_date="01-2000")
        assert len(out) == len(df)


class TestLoadFilteredForecastMonth:
    """
    On the ``forecast_month`` layout the filter is a string comparison, which
    is only correct because the column is written zero-padded ('2026-07').
    """

    def test_drops_earlier_months_keeps_later(self, tmp_path):
        """Rows before the threshold are excluded."""
        out = _write(tmp_path, _forecast_month_frame()).load(start_date="07-2026")
        assert set(out["forecast_month"]) == {"2026-07", "2026-10", "2027-01"}

    def test_single_digit_month_sorts_correctly(self, tmp_path):
        """
        '2026-10' >= '2026-07' must hold as a string compare. Unpadded values
        ('2026-7') would break this ordering, so this guards the zero-padding
        assumption the query relies on.
        """
        out = _write(tmp_path, _forecast_month_frame()).load(start_date="07-2026")
        assert "2026-10" in set(out["forecast_month"])


class TestLoadFilteredErrors:
    """``start_date`` requires a recognizable date layout."""

    def test_raises_when_no_date_columns(self, tmp_path):
        """A table with neither layout cannot be filtered."""
        df = pd.DataFrame({"cfs_run": ["2026050100"], "value": [1.0]})
        db = _write(tmp_path, df, table="odd_table")
        with pytest.raises(ValueError, match="forecast_month"):
            db.load(start_date="07-2026")

    def test_start_date_is_bound_as_a_parameter(self, tmp_path):
        """
        start_date reaches SQLite as a bound parameter, not interpolated SQL,
        so a string with quotes cannot alter the query. pandas.Period rejects
        it before it ever gets that far.
        """
        db = _write(tmp_path, _year_month_frame())
        with pytest.raises(Exception):
            db.load(start_date="2026-07'; DROP TABLE cnbs_forecast; --")
        # The table is still intact either way.
        assert len(db.load()) == len(_year_month_frame())


# ===========================================================================
# Equivalence: SQL pushdown must match filtering in pandas
# ===========================================================================
class TestPushdownMatchesPandasFilter:
    """
    ``load(start_date=X)`` must return exactly what
    ``ForecastTransformer(load()).filter(X)`` returns. This is the property
    that lets notebook 3 push the filter into SQL without changing outputs.
    """

    @staticmethod
    def _assert_same(pushed, in_pandas, key):
        """
        Compare the two paths on rows, columns and values.

        ``check_dtype=False`` because an empty SQL result comes back with
        ``object`` columns, while filtering a populated frame down to nothing
        in pandas preserves the original dtypes. That divergence exists only
        for empty results and reflects how ``read_sql`` types a zero-row
        result, not a difference in what gets selected.
        """
        pd.testing.assert_frame_equal(
            pushed.sort_values(key).reset_index(drop=True),
            in_pandas.sort_values(key).reset_index(drop=True),
            check_dtype=False,
        )

    @pytest.mark.parametrize("start_date", ["05-2026", "07-2026", "01-2027", "01-2030"])
    def test_year_month_layout_matches(self, tmp_path, start_date):
        """Equivalence across thresholds below, inside, and past the data range."""
        db = _write(tmp_path, _year_month_frame())
        self._assert_same(
            db.load(start_date=start_date),
            ForecastTransformer(db.load()).filter(start_date),
            ["year", "month"],
        )

    @pytest.mark.parametrize("start_date", ["05-2026", "07-2026", "01-2027"])
    def test_forecast_month_layout_matches(self, tmp_path, start_date):
        """Same equivalence on the forecast_month layout."""
        db = _write(tmp_path, _forecast_month_frame())
        self._assert_same(
            db.load(start_date=start_date),
            ForecastTransformer(db.load()).filter(start_date),
            ["forecast_month"],
        )

    def test_ensemble_members_all_survive(self, tmp_path):
        """
        Every ensemble member of a kept month is returned — the filter selects
        whole months, it does not collapse rows that share a (year, month).
        """
        db = _write(tmp_path, _ensemble_frame())
        out = db.load(start_date="07-2026")

        per_month = out.groupby(["year", "month"]).size().to_dict()
        assert per_month == {(2026, 7): 24, (2027, 1): 24}

    def test_ensemble_layout_matches_pandas_filter(self, tmp_path):
        """Equivalence holds on ensemble-shaped data, duplicates included."""
        db = _write(tmp_path, _ensemble_frame())
        self._assert_same(
            db.load(start_date="07-2026"),
            ForecastTransformer(db.load()).filter("07-2026"),
            ["cfs_run", "year", "month", "model", "lake", "component"],
        )

    def test_dtypes_match_when_rows_survive(self, tmp_path):
        """
        For any non-empty result the two paths agree on dtypes too, so the
        relaxation above is genuinely confined to the empty case.
        """
        db = _write(tmp_path, _year_month_frame())
        pushed = db.load(start_date="07-2026")
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
