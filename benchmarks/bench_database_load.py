"""Benchmark and verify the CNBS database loading path.

Reproduces the measurements behind the SQL-filtering change for notebook 3
(Visuals), which was slow for USACE users reading ``cnbs_forecast.db`` from a
shared network drive.

It builds a synthetic, ensemble-shaped forecast table in a temporary directory,
then compares three paths:

1. **legacy** — ``load()`` then a per-row ``pd.Period`` filter (the original
   implementation, reproduced here for reference).
2. **vectorized** — ``load()`` then the current vectorized filter.
3. **pushdown** — ``load(start_date=...)``, filtering in SQL.

Alongside the timings it asserts the three paths return *identical* rows, and
prints the SQLite query plan so you can see whether the filtered query uses the
date index or still scans the table.

For actual page-read counts — the closest proxy for bytes pulled over a network
share — use the ``sqlite3`` CLI, which exposes the counters the Python module
does not::

    sqlite3 <db> ".stats on" "SELECT * FROM cnbs_forecast WHERE year >= 2027;"

Run it directly; nothing is written outside a temporary directory::

    conda activate nbs_env_test
    python benchmarks/bench_database_load.py            # ~2M rows, a few minutes
    python benchmarks/bench_database_load.py --small    # ~200k rows, quick

Note on interpreting the results: the ``WHERE`` clause only reduces bytes read
if the date columns are indexed. Without the index SQLite scans every page and
reads just as much as an unfiltered query — which is why
``CFSDatabase.add_df`` now indexes after each write.
"""

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import pandas as pd

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import ForecastTransformer  # noqa: E402
from src.database_utils import CFSDatabase  # noqa: E402

TABLE = "cnbs_forecast"
START_DATE = "07-2026"

LAKES = ["superior", "michigan-huron", "erie", "ontario"]
COMPONENTS = ["precipitation", "evaporation", "runoff", "nbs"]
MODELS = ["RF", "GP", "XGB", "MLR", "SVR"]


def legacy_filter(df, first_forecast_month):
    """
    The original ``ForecastTransformer.filter`` year/month branch.

    Kept here verbatim so the benchmark can show what the change replaced. It
    builds one :class:`pandas.Period` per row via ``df.apply``, which is what
    made notebook 3 slow.
    """
    df = df.copy()
    min_period = pd.Period(first_forecast_month, freq="M")
    df["forecast_period"] = df.apply(
        lambda r: pd.Period(f"{r.year}-{r.month:02d}", freq="M"), axis=1
    )
    first_month_to_keep = max(df["forecast_period"].min(), min_period)
    df = df[df["forecast_period"] >= first_month_to_keep]
    return df.drop(columns=["forecast_period"])


def build_table(database, months_of_runs=18, leads=12):
    """
    Write a synthetic forecast table shaped like the operational one.

    Every forecast month carries a full ensemble — one row per
    (cfs_run, model, lake, component) — so ``(year, month)`` is heavily
    non-unique, as in the real database.

    Parameters
    ----------
    database : str
        Path of the SQLite file to create.
    months_of_runs : int, default 18
        How many months of 6-hourly CFS runs to generate.
    leads : int, default 12
        Forecast months per run.

    Returns
    -------
    pandas.DataFrame
        The frame that was written.
    """
    end = pd.Timestamp("2026-06-30")
    start = end - pd.DateOffset(months=months_of_runs)
    runs = pd.date_range(start, end, freq="6h")

    rows = []
    for run in runs:
        run_id = run.strftime("%Y%m%d%H")
        for lead in range(leads):
            valid = run + pd.DateOffset(months=lead)
            for model in MODELS:
                for lake in LAKES:
                    for component in COMPONENTS:
                        rows.append((
                            run_id, valid.month, valid.year,
                            model, lake, component, 1.0, 2.0,
                        ))

    df = pd.DataFrame(rows, columns=[
        "cfs_run", "month", "year", "model", "lake", "component",
        "value [mm]", "value [cms]",
    ])

    # add_df indexes the date columns as part of the write.
    CFSDatabase(database, TABLE).add_df(df)
    return df


def time_it(fn):
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0


def query_plan(database, sql):
    """Return the EXPLAIN QUERY PLAN description lines for ``sql``."""
    conn = sqlite3.connect(database)
    try:
        conn.execute("ANALYZE")
        return [row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}")]
    finally:
        conn.close()


def index_names(database):
    """Return user-defined index names in the database."""
    conn = sqlite3.connect(database)
    try:
        return [
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        ]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--small", action="store_true",
        help="Use a ~200k-row table instead of ~2M for a quick run.",
    )
    args = parser.parse_args()

    months = 2 if args.small else 18
    workdir = tempfile.mkdtemp(prefix="cnbs_bench_")
    database = os.path.join(workdir, "cnbs_forecast.db")

    try:
        print("Building synthetic forecast table...")
        df, build_s = time_it(lambda: build_table(database, months_of_runs=months))
        size_mb = os.path.getsize(database) / 1e6
        print(f"  rows      : {len(df):,}")
        print(f"  file size : {size_mb:.1f} MB  (includes the date index)")
        print(f"  build time: {build_s:.1f}s")
        print(f"  indexes   : {', '.join(index_names(database)) or 'none'}")

        db = CFSDatabase(database, TABLE)

        print(f"\nFiltering from {START_DATE}...\n")

        full, t_load = time_it(db.load)
        legacy, t_legacy = time_it(lambda: legacy_filter(full, START_DATE))
        vector, t_vector = time_it(
            lambda: ForecastTransformer(full).filter(START_DATE)
        )
        pushed, t_push = time_it(lambda: db.load(start_date=START_DATE))
        pushed_f, t_push_f = time_it(
            lambda: ForecastTransformer(pushed).filter(START_DATE)
        )

        print(f"{'path':<34}{'load':>9}{'filter':>10}{'total':>10}")
        print("-" * 63)
        print(f"{'legacy (per-row Period)':<34}{t_load:8.2f}s{t_legacy:9.2f}s"
              f"{t_load + t_legacy:9.2f}s")
        print(f"{'vectorized filter only':<34}{t_load:8.2f}s{t_vector:9.2f}s"
              f"{t_load + t_vector:9.2f}s")
        print(f"{'pushdown + vectorized':<34}{t_push:8.2f}s{t_push_f:9.2f}s"
              f"{t_push + t_push_f:9.2f}s")

        before = t_load + t_legacy
        after = t_push + t_push_f
        print(f"\nend-to-end speedup: {before / after:.0f}x "
              f"({before:.1f}s -> {after:.1f}s)")

        # --- correctness: all three paths must agree ---
        key = ["cfs_run", "year", "month", "model", "lake", "component"]

        def norm(frame):
            return frame.sort_values(key).reset_index(drop=True)

        legacy_n, vector_n, pushed_n = norm(legacy), norm(vector), norm(pushed_f)
        print(f"\nrows kept: {len(pushed_n):,} of {len(df):,}")
        print(f"  legacy == vectorized : {legacy_n.equals(vector_n)}")
        print(f"  legacy == pushdown   : {legacy_n.equals(pushed_n)}")
        assert legacy_n.equals(vector_n), "vectorized filter changed the result"
        assert legacy_n.equals(pushed_n), "SQL pushdown changed the result"

        # Ensembles: whole months are selected, members are never collapsed.
        members = pushed_n.groupby(["year", "month"]).size().unique()
        print(f"  ensemble members per kept month: {sorted(members)}")

        # --- I/O: does the filtered query use the index? ---
        filtered_sql = (
            f"SELECT * FROM {TABLE} "
            "WHERE year > 2026 OR (year = 2026 AND month >= 7)"
        )
        print("\nquery plan (filtered):")
        for line in query_plan(database, filtered_sql):
            print(f"  {line}")
        print("\nA plan containing 'USING INDEX' is what reduces bytes read over")
        print("a network share. 'SCAN' means the full table is being read --")
        print("expected on small or unselective tables, where SQLite correctly")
        print("judges a scan cheaper than an index lookup.")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
