"""SQLite storage for processed CFS forecast data.

Wraps a single SQLite database/table behind :class:`CFSDatabase`, providing
schema creation, row- and DataFrame-level inserts, point lookups, and helpers
for figuring out which forecast run to download next. The schema keys each
value by ``(cfs_run, year, month, lake, surface_type, component)``.

The forecast notebooks use this module to persist processed CFS output and to
resume incremental downloads where the previous run left off.
"""

import sqlite3
import os
import pandas as pd
import sys
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

class CFSDatabase:
    """Manage a SQLite database of processed CFS forecast values.

    Each instance is bound to one database file and one table. The table
    stores one value per ``(cfs_run, year, month, lake, surface_type,
    component)`` key. Methods cover schema creation, inserting individual
    records or whole DataFrames, point lookups, and determining the next CFS
    run to download.

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
        Ensure database file exists.
        """
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

    def load(self):
        """
        Load the entire table into a DataFrame.

        Returns
        -------
        pandas.DataFrame
            All rows of the configured table.
        """
        # Create a connection to the SQLite database
        conn = sqlite3.connect(self.database)

        # Use an f-string to insert the table name properly
        query = f'SELECT * FROM {self.table}'

        # Execute the query and fetch the data into a DataFrame
        data = pd.read_sql(query, conn)

        # Close the connection once done
        conn.close()

        return data

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
                    conn.execute("PRAGMA journal_mode=WAL;")  # allows concurrent reads
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
        Add an entire pandas DataFrame to the database table.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to insert. Must contain the columns
            ``['cfs_run', 'forecast_month', 'model', 'lake', 'precipitation',
            'evaporation', 'runoff', 'nbs']``.
        if_exists : str, default "append"
            How to behave if the table already exists. One of ``'fail'``,
            ``'replace'``, or ``'append'``.
        """

        required_cols = ['cfs_run', 'forecast_month', 'model', 'lake', 'precipitation', 'evaporation', 'runoff', 'nbs']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

        try:
            conn = sqlite3.connect(self.database)

            # Insert DataFrame into the database
            df.to_sql(self.table, conn, if_exists=if_exists, index=False)

            conn.close()
        except sqlite3.DatabaseError as e:
            raise sqlite3.DatabaseError(f"Database error occurred while inserting DataFrame: {e}")
        
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