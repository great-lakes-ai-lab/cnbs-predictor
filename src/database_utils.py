"""SQLite storage for processed CFS forecast data.

Wraps a single SQLite database/table behind :class:`CFSDatabase`, providing
schema creation, row- and DataFrame-level inserts, point lookups, and helpers
for figuring out which forecast run to download next. The schema keys each
value by ``(cfs_run, year, month, lake, surface_type, component)``.

The forecast notebooks use this module to persist processed CFS output and to
resume incremental downloads where the previous run left off.
"""

import sqlite3
from pathlib import Path
import os
import pandas as pd
import sys
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

class Database:
    """Manage a SQLite database of processed forecast values.

    Parameters
    ----------
    database : str
        Path to the SQLite database file (created if it does not exist).
    table : str
        Name of the table to read from and write to.
    """

    def __init__(self, database, table):
        """
        Initialize database connection and ensure table exists.
        """
        self.database = database
        self.table = table
        self._initialize_database()

    def _initialize_database(self):
        """
        Ensure database directory and file exist.
        """

        # Create parent directory if it does not exist
        database_path = Path(self.database)

        if database_path.parent != Path("."):
            database_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

        # SQLite will create the database file if it does not exist
        conn = sqlite3.connect(self.database)
        conn.close()

    def create_cfs_table(self):
        """
        Create the standard CFS table schema if it does not exist.
        """
        with sqlite3.connect(self.database) as conn:
            conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table} (
                    cfs_run TEXT,
                    year INTEGER,
                    month INTEGER,
                    lake TEXT,
                    surface_type TEXT,
                    component TEXT,
                    value REAL,
                    PRIMARY KEY (cfs_run, year, month, lake, surface_type, component)
                )
            ''')

    def _date_columns(self, conn):
        """
        Report which forecast-date columns the table actually has.

        The two writers in notebook 2 produce different layouts: the pivoted
        frame carries a single ``forecast_month`` ('YYYY-MM') column, while the
        unit-converted frame carries separate integer ``year`` and ``month``
        columns. Callers sniff the layout the same way :meth:`pull` and
        :meth:`add` sniff ``value`` vs ``value [mm]``.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open connection to the database.

        Returns
        -------
        tuple of bool
            ``(has_year_month, has_forecast_month)``.
        """
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info({self.table})')}
        return {"year", "month"} <= columns, "forecast_month" in columns

    def load(
        self,
        start_date=None,
        date_column=None,
        date_column_format="%m-%Y"
    ):
        """
        Load the table into a DataFrame, optionally filtering by start date.

        Parameters
        ----------
        start_date : str, optional
            Start date in either:
                "%m-%Y"
                "%m-%d-%Y"

            Examples:
                "09-2026"
                "09-15-2026"

        date_column : str or list/tuple of str, optional
            Column containing the date to filter on.

            For separate year/month columns:
                ["year", "month"]

        date_column_format : str, default="%m-%Y"
            Format of the date stored in the database column.

            Examples:
                "%Y-%m"
                "%Y-%m-%d"
                "%m/%d/%Y"

        Returns
        -------
        pandas.DataFrame
            Loaded data.
        """

        conn = sqlite3.connect(self.database)

        try:

            # --------------------------------------------------------
            # Load full table
            # --------------------------------------------------------

            query = f'SELECT * FROM "{self.table}"'

            data = pd.read_sql(
                query,
                conn
            )

        finally:
            conn.close()

        # ------------------------------------------------------------
        # No start date = return entire database
        # ------------------------------------------------------------

        if start_date is None:
            return data

        # ------------------------------------------------------------
        # Validate date column
        # ------------------------------------------------------------

        if date_column is None:
            raise ValueError(
                "date_column must be specified when start_date is provided."
            )

        # ------------------------------------------------------------
        # Parse start_date
        # Accept either MM-YYYY or MM-DD-YYYY
        # ------------------------------------------------------------

        try:
            start = pd.to_datetime(
                start_date,
                format="%m-%Y"
            )

        except ValueError:

            try:
                start = pd.to_datetime(
                    start_date,
                    format="%m-%d-%Y"
                )

            except ValueError:
                raise ValueError(
                    f"Invalid start_date '{start_date}'. "
                    "Expected format '%m-%Y' or '%m-%d-%Y'."
                )

        # ------------------------------------------------------------
        # Separate year/month columns
        # ------------------------------------------------------------

        if isinstance(date_column, (list, tuple)):

            if len(date_column) != 2:
                raise ValueError(
                    "When using multiple date columns, provide "
                    "exactly two columns: ['year', 'month']."
                )

            year_column, month_column = date_column

            if year_column not in data.columns:
                raise ValueError(
                    f"Column '{year_column}' does not exist."
                )

            if month_column not in data.columns:
                raise ValueError(
                    f"Column '{month_column}' does not exist."
                )

            data["_filter_date"] = pd.to_datetime(
                data[year_column].astype(str)
                + "-"
                + data[month_column].astype(str).str.zfill(2)
                + "-01"
            )

            data = data[
                data["_filter_date"] >= start
            ].drop(
                columns="_filter_date"
            )

        # ------------------------------------------------------------
        # Single date column
        # ------------------------------------------------------------

        else:

            if date_column not in data.columns:
                raise ValueError(
                    f"Column '{date_column}' does not exist."
                )

            data["_filter_date"] = pd.to_datetime(
                data[date_column],
                format=date_column_format
            )

            data = data[
                data["_filter_date"] >= start
            ].drop(
                columns="_filter_date"
            )

        # ------------------------------------------------------------
        # Reset index
        # ------------------------------------------------------------

        data = data.reset_index(drop=True)

        return data

    def create_indexes(self):
        """
        Index the table's forecast-date columns, if it has any.

        Creating the index is what makes ``load(start_date=...)`` read less
        data: with no index SQLite scans the full table even when a WHERE
        clause is present, so a filtered query pulls just as many bytes over a
        network share as an unfiltered one.

        Safe to call repeatedly — it uses ``CREATE INDEX IF NOT EXISTS`` and
        skips whichever columns the table does not have. :meth:`add_df` calls
        it after every write, which also restores the index after an
        ``if_exists='replace'`` write drops the table along with its indexes.

        Returns
        -------
        list of str
            Names of the indexes that now exist for this table.
        """
        created = []

        conn = sqlite3.connect(self.database)
        try:
            has_year_month, has_forecast_month = self._date_columns(conn)

            if has_year_month:
                name = f'idx_{self.table}_year_month'
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS {name} ON {self.table} (year, month)'
                )
                created.append(name)

            if has_forecast_month:
                name = f'idx_{self.table}_forecast_month'
                conn.execute(
                    f'CREATE INDEX IF NOT EXISTS {name} ON {self.table} (forecast_month)'
                )
                created.append(name)

            conn.commit()
        finally:
            conn.close()

        return created

    def pull(self, cfs_run, year, month, lake, surface_type, component):
        """
        Look up a single stored value by its full primary key.

        Detects whether the table uses a ``value`` or ``value [mm]`` column and
        queries accordingly.

        Parameters
        ----------
        cfs_run : str
            CFS run identifier (YYYYMMDDHH).
        year : int
            Forecast year.
        month : int
            Forecast month (1-12).
        lake : str
            Lake name.
        surface_type : str
            Surface type ('lake' or 'land').
        component : str
            NBS component ('precipitation', 'evaporation', 'runoff', 'nbs').

        Returns
        -------
        float or None
            The stored value, or None if no matching row exists or a database
            error occurs.
        """
        try:
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            # --- Detect which "value" column exists ---
            cursor.execute(f"PRAGMA table_info({self.table})")
            columns = [row[1] for row in cursor.fetchall()]
            if "value [mm]" in columns:
                value_col = '"value [mm]"'
            elif "value" in columns:
                value_col = "value"
            else:
                raise RuntimeError(
                    f"Neither 'value' nor 'value [mm]' column found in table '{self.table}'."
                )

            # --- Build query using detected column name ---
            query = f'''
            SELECT {value_col} FROM {self.table}
            WHERE cfs_run = ? AND year = ? AND month = ? 
                AND lake = ? AND surface_type = ? AND component = ?
            '''
            cursor.execute(query, (cfs_run, year, month, lake, surface_type, component))
            result = cursor.fetchone()
            conn.close()

            if result:
                return result[0]
            else:
                print(f"No data found for {locals()}")
                return None

        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None

    def add(self, cfs_run, year, month, lake, surface_type, component, value):
        """
        Safely adds or replaces a record in the database table using SQLite.

        Parameters:
            cfs_run (str): The CFS run identifier (YYYYMMDDHH).
            year (int): Forecast year (e.g., 2024, 2025).
            month (int): Forecast month (1–12).
            lake (str): Lake name.
            surface_type (str): Surface type ('lake' or 'land').
            component (str): NBS Component ('precipitation', 'evaporation', 'runoff', 'cnbs').
            value (float): Data value.

        Raises:
            ValueError: For invalid parameter types or ranges.
            sqlite3.DatabaseError: For database interaction errors.
        """

        # Ensure the table exists before attempting to add data
        self.create_cfs_table()

        # --- Input validation ---
        if not isinstance(year, int):
            raise ValueError("ERROR: Year must be an integer.")
        if not (1 <= month <= 12):
            raise ValueError("ERROR: Month must be between 1 and 12.")
        if not all(isinstance(v, str) for v in [cfs_run, lake, surface_type, component]):
            raise ValueError("ERROR: cfs_run, lake, surface_type, and component must be strings.")
        if not isinstance(value, (float, int)):
            raise ValueError("ERROR: Value must be numeric.")

        max_retries = 5
        retry_delay = 3  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                # Use context manager for safe connection handling
                with sqlite3.connect(self.database, timeout=30) as conn:
                    #conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent reads
                    cursor = conn.cursor()

                    # Get table columns to find correct value column
                    cursor.execute(f"PRAGMA table_info({self.table})")
                    columns = [row[1].strip() for row in cursor.fetchall()]

                    if "value [mm]" in columns:
                        value_col = '"value [mm]"'
                    elif "value" in columns:
                        value_col = "value"
                    else:
                        raise RuntimeError(
                            f"Neither 'value' nor 'value [mm]' column found in table '{self.table}'"
                        )

                    # Insert or replace record
                    query = f"""
                    INSERT OR REPLACE INTO {self.table} (
                        cfs_run, year, month, lake, surface_type, component, {value_col}
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(query, (cfs_run, year, month, lake, surface_type, component, value))
                    conn.commit()
                return  # ✅ success — exit early

            except sqlite3.OperationalError as e:
                # Handle transient locking or “readonly” issues
                if "locked" in str(e).lower() or "readonly" in str(e).lower():
                    print(f"Database busy or locked (attempt {attempt}/{max_retries}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise sqlite3.DatabaseError(f"Database error occurred: {e}")
            except sqlite3.DatabaseError as e:
                raise sqlite3.DatabaseError(f"Database error occurred: {e}")

        print(f"Failed to write record after {max_retries} attempts — skipping forecast.")

        
    def add_df(self, df, if_exists="append"):
        """
        Add a pandas DataFrame to the database table.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to insert.
        if_exists : {"append", "replace", "fail"}, default "append"
            Behavior if the table already exists.

        Notes
        -----
        Indexes the forecast-date columns afterwards via
        :meth:`create_indexes`, so readers get the benefit of
        ``load(start_date=...)`` without having to index the database
        themselves. If indexing fails the write is still kept and a warning is
        printed, since the index only affects speed and not correctness.
        """

        try:
            with sqlite3.connect(self.database) as conn:

                # Check if the table already exists
                table_exists = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (self.table,)
                ).fetchone() is not None

                # If appending to an existing table, verify columns match
                if table_exists and if_exists == "append":
                    db_columns = pd.read_sql_query(
                        f"SELECT * FROM {self.table} LIMIT 0",
                        conn
                    ).columns.tolist()

                    df_columns = df.columns.tolist()

                    if db_columns != df_columns:
                        raise ValueError(
                            "DataFrame columns do not match database table.\n"
                            f"Database:  {db_columns}\n"
                            f"DataFrame: {df_columns}"
                        )

                # Write DataFrame
                df.to_sql(self.table, conn, if_exists=if_exists, index=False)

        except sqlite3.DatabaseError as e:
            raise sqlite3.DatabaseError(
                f"Database error occurred while inserting DataFrame: {e}"
            )

        # Keep the date index in place for readers using load(start_date=...).
        # Done after the write completes: a "replace" write drops the table and
        # its indexes, so re-creating here is what restores them.
        #
        # Indexing is an optimization, so failing to index must not turn a
        # committed write into a raised error — the caller would otherwise
        # retry and append the rows a second time. Warn and carry on instead;
        # queries stay correct without the index, just slower.
        try:
            self.create_indexes()
        except sqlite3.DatabaseError as e:
            print(
                f"WARNING: data was written, but indexing table '{self.table}' "
                f"failed: {e}. Reads will still be correct, but "
                "load(start_date=...) will not be able to skip rows."
            )
        
    def get_next_run(self):
        """
        Determine the next CFS run date to download.

        Reads the most recent ``cfs_run`` in the table and returns the date
        from which downloading should resume: the same date at midnight if the
        last run was the 00/06/12 cycle, or the following day if it was the 18
        cycle. If the table is missing or empty, falls back to the first of the
        month nine months ago.

        Returns
        -------
        datetime.datetime
            The next run date, with the time component zeroed.
        """
        try:
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()

            # Check if the table exists
            cursor.execute(f'''
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='{self.table}'
            ''')
            if not cursor.fetchone():
                raise sqlite3.OperationalError(f"Table '{self.table}' does not exist. Resorting to fallback date.")

            # Get the most recent run
            cursor.execute(f'''
                SELECT cfs_run FROM {self.table} 
                ORDER BY cfs_run DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                raw_value = str(result[0]).strip()

                # Try parsing with common formats
                for fmt in ("%Y%m%d%H", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H"):
                    try:
                        last_run = datetime.strptime(raw_value, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError(f"Unrecognized datetime format: {raw_value}")

                hour = last_run.hour

                # Decide what to return based on the hour
                if hour in (0, 6, 12):
                    # Same date at midnight
                    next_run = last_run.replace(hour=0, minute=0, second=0, microsecond=0)
                    print(f"WARNING: Last date {next_run.strftime('%m-%d-%Y')} did not download all runs. Redownloading beginning from that date.")
                elif hour == 18:
                    # Next day at midnight
                    next_run = (last_run + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                else:
                    # Unexpected hour - fallback behavior, treat as same date at midnight with warning
                    next_run = last_run.replace(hour=0, minute=0, second=0, microsecond=0)
                    print(f"WARNING: Unexpected hour {hour} in last run date. Defaulting to {next_run.strftime('%m-%d-%Y')} at midnight.")

                return next_run  # datetime object with time zeroed

            else:
                raise ValueError("Table exists but no previous run was found. Resorting to fallback date.")

        except (sqlite3.Error, ValueError) as e:
            print(f"WARNING: {e}")
            # Fallback: first of the month, 9 months ago at midnight UTC
            now_utc = datetime.utcnow().replace(tzinfo=None)
            nine_months_ago = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0) - relativedelta(months=9)
            return nine_months_ago  # datetime object at midnight

    def get_date_range(self, auto, start_date, end_date):
        """
        Determine the start and end dates for CFS CSV downloads, and return the date range.

        Parameters
        ----------
        auto : str,
            Whether to automatically fetch the next run date from the database ('yes' or 'no').
        start_date : str,
            Manual start date in format 'MM-DD-YYYY'. Required if auto='no'.
        end_date : str,
            Manual end date in format 'MM-DD-YYYY'. Required if auto='no'.

        Returns
        -------
        tuple
            (start_date: datetime, end_date: datetime, date_array: pd.DatetimeIndex)
        """
        if auto.lower() == 'yes':
            start_date = self.get_next_run()
            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif auto.lower() == 'no':
            start_date = datetime.strptime(start_date, "%m-%d-%Y")
            end_date = datetime.strptime(end_date, "%m-%d-%Y")
        else:
            raise ValueError("Invalid value for 'auto'. Please enter 'yes' or 'no'.")
        
        # Validate dates
        if start_date == end_date:
            print("The CSV files are up-to-date. Script will exit now.")
            sys.exit(0)
        elif start_date > end_date:
            raise ValueError("End date cannot be older than start date. Try again.")
        else:
            print(f"Starting from: {start_date.strftime('%m-%d-%Y')} and continuing through: {end_date.strftime('%m-%d-%Y')}")

        date_array = pd.date_range(start=start_date, end=end_date, freq='1d')
        return start_date, end_date, date_array

    def print_columns(self):
        """
        Print the name and declared type of each column in the table.

        Intended as an interactive/debugging helper; prints a message if the
        table does not exist or has no columns.
        """
        try:
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()
            cursor.execute(f'PRAGMA table_info({self.table})')
            columns = cursor.fetchall()
            conn.close()

            if columns:
                print(f"Columns in table '{self.table}':")
                for col in columns:
                    print(f"- {col[1]} ({col[2]})")
            else:
                print(f"Table '{self.table}' does not exist or has no columns.")

        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def create_sfs_table(self):
            """
            Create the standard CFS table schema if it does not exist.
            """
            with sqlite3.connect(self.database) as conn:
                conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        init_time TEXT,
                        member INTEGER,
                        lead INTEGER,
                        valid_time TEXT,
                        lake TEXT,
                        surface_type TEXT,
                        component TEXT,
                        value REAL,
                        PRIMARY KEY (init_time, member, lead, valid_time, lake, surface_type, component)
                    )
                ''')