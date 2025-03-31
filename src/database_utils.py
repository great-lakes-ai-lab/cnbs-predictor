import urllib.request
from bs4 import BeautifulSoup
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
import sqlite3
from datetime import datetime, timedelta

def download_grb2_ncei(product, url_path, download_dir):
    """
    Downloads GRB2 CFS forecast files from the National Centers for Environmental Information (NCEI).

    Parameters:
    product (str): The product type (e.g., 'pgbf', 'flxf') used to filter GRB2 files.
    url_path (str): The URL path to the NCEI directory containing the GRB2 files.
    download_dir (str): The local directory where the files should be downloaded.

    Returns:
    None
    """
    # Input validation
    if not isinstance(product, str):
        raise ValueError("ERROR: The 'product' parameter must be a string.")
    
    if not isinstance(url_path, str):
        raise ValueError("ERROR: The 'url_path' parameter must be a string representing the NCEI URL path.")
    
    if not isinstance(download_dir, str):
        raise ValueError("ERROR: The 'download_dir' parameter must be a string representing the local download directory.")
    
    if not os.path.exists(download_dir):
        raise ValueError(f"ERROR: The specified download directory does not exist: {download_dir}")

    try:
        # Open the URL and parse the HTML content
        response = urllib.request.urlopen(url_path)
        html_content = response.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all links that match the product type and end with 'grib.grb2'
        links = soup.find_all('a', href=lambda href: href and href.startswith(product) and href.endswith('grib.grb2'))
        
        # Download the files
        for link in links:
            file_url = url_path + link['href']
            filename = link['href'].split('/')[-1]
            file_path = os.path.join(download_dir, filename)
            
            # Download the file
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded: {filename}")

    except urllib.request.URLError as e:
        print(f"ERROR with the URL request: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

def download_grb2_aws(product, bucket_name, url_path, download_dir):
    """
    Downloads GRB2 CFS forecast files from Amazon Web Services (AWS).

    Parameters:
    - product (str): 'flx' or 'pgb' (specifies the product type)
    - bucket_name (str): The name of the S3 bucket (e.g., 'noaa-cfs-pds' for CFS data)
    - url_path (str): The S3 URL path to the data within the bucket
    - download_dir (str): Local directory where the downloaded data should be stored
    """
    # Input validation
    if not isinstance(product, str):
        raise ValueError("ERROR: The 'product' parameter must be a string.")
    
    if not isinstance(bucket_name, str):
        raise ValueError("ERROR: The 'bucket_name' parameter must be a string.")
    
    if not isinstance(url_path, str):
        raise ValueError("ERROR: The 'url_path' parameter must be a string representing the S3 URL path.")
    
    if not isinstance(download_dir, str):
        raise ValueError("ERROR: The 'download_dir' parameter must be a string representing the local download directory.")
    
    if not os.path.exists(download_dir):
        raise ValueError(f"ERROR: The specified download directory does not exist: {download_dir}")

    # Create a boto3 client for S3 with unsigned requests (no AWS credentials needed)
    s3_config = Config(signature_version='UNSIGNED')
    s3 = boto3.client('s3', config=s3_config)

    try:
        # List all objects in the specified folder path, handling pagination if needed
        continuation_token = None
        objects = []

        while True:
            list_objects_args = {'Bucket': bucket_name, 'Prefix': url_path}
            if continuation_token:
                list_objects_args['ContinuationToken'] = continuation_token

            list_objects_response = s3.list_objects_v2(**list_objects_args)
            objects.extend(list_objects_response.get('Contents', []))

            if not list_objects_response.get('IsTruncated', False):
                break

            continuation_token = list_objects_response.get('NextContinuationToken')

        # Iterate over each object and download if it matches the product and ends with 'grib.grb2'
        for obj in objects:
            key = obj['Key']
            if product in key and key.endswith('grib.grb2'):
                local_file_path = os.path.join(download_dir, os.path.relpath(key, url_path))

                # Ensure the directory structure exists
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the file
                s3.download_file(bucket_name, key, local_file_path)
                print(f"Downloaded: {key}")

    except NoCredentialsError:
        print("ERROR: No AWS credentials were found. Ensure that your environment is properly configured.")
    except PartialCredentialsError:
        print("ERROR: Incomplete AWS credentials found. Please ensure all necessary credentials are provided.")
    except Exception as e:
        print(f"ERROR: {e}")

def check_url_exists(url):
    """
    Check if a URL exists by sending a HEAD request.

    Parameters:
    - url (str): The URL to check.

    Returns:
    - bool: True if the URL returns a status code 200 (OK), False otherwise.
    """
    try:
        response = requests.head(url, allow_redirects=True)  # Allow redirects in case of URL redirection
        # Check if the response is OK (status code 200)
        return response.status_code == 200
    except requests.RequestException as e:
        # You could log the error or handle specific exceptions (e.g., network issues)
        print(f"ERROR occurred while checking the URL: {e}")
        return False
    

def open_cfs_db(database):
    """
    Opens a connection to the database. If the database does not exist, it creates a new one.
    It also creates a table `forecast_data` if it does not already exist.

    Parameters:
    - database (str): The path to the SQLite database file.

    Returns:
    - conn (sqlite3.Connection): The connection object to the database.
    - cursor (sqlite3.Cursor): The cursor object to execute SQL commands.
    """
    try:
        # Check if the database file exists
        if not os.path.exists(database):
            print(f"Creating new database: '{database}'.")

        # Connect to the database (it will create it if it doesn't exist)
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Create the forecast_data table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_data (
            cfs_run INTEGER,
            year INTEGER,
            month INTEGER,
            lake TEXT,
            surface_type TEXT,
            cnbs TEXT,
            value REAL,
            PRIMARY KEY (cfs_run, year, month, lake, surface_type, cnbs)
        )
        ''')

        # Commit the changes (though nothing to commit here since it's a table creation)
        conn.commit()
        
        # Return connection and cursor
        return conn, cursor

    except sqlite3.Error as e:
        print(f"ERROR opening/creating database: {e}")
        return None, None


def pull_from_db(database, table, cfs_run, year, month, lake, surface_type, cnbs):
    """
    Pulls a value from the database based on specific query parameters.

    Parameters:
    - database (str): Path to database.
    - table (str): Table name from which to fetch the data.
    - cfs_run (int): The forecast run identifier.
    - year (int): The forecast year.
    - month (int): The forecast month.
    - lake (str): The lake name.
    - surface_type (str): The type of surface ('land' or 'lake').
    - cnbs (str): The CNBS type (e.g., 'precipitation', 'evaporation').

    Returns:
    - value (float or None): The value for the specified query parameters, or None if not found or an error occurs.
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # SQL query to fetch the value from the specified table
        query = f'''
        SELECT value FROM {table}
        WHERE cfs_run = ? AND year = ? AND month = ? AND lake = ? AND surface_type = ? AND cnbs = ?
        '''

        # Execute the query with the provided parameters
        cursor.execute(query, (cfs_run, year, month, lake, surface_type, cnbs))

        # Fetch the result (None if not found)
        result = cursor.fetchone()

        # Close the database connection
        conn.close()

        # Return the value if found, otherwise return None
        if result:
            return result[0]
        else:
            print(f"ERROR: No data found for cfs_run={cfs_run}, year={year}, month={month}, lake={lake}, surface_type={surface_type}, cnbs={cnbs}")
            return None

    except sqlite3.Error as e:
        print(f"ERROR accessing the database: {e}")
        return None
    
def get_next_cfs_run(database, table):
    """
    Retrieves the next CFS run timestamp from the database table based on the last entry.
    The function calculates the next forecast run time, which is incremented by 6 hours.

    Parameters:
    - database (str): Path to the SQLite database file.
    - table (str): Name of the table containing the CFS run data.

    Returns:
    - next_cfs_run (str): The next CFS run time in MM-DD-YYYY HH format, or None if an error occurs.
    """
    try:
        # Establish a connection to the SQLite database
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Query to get the last entry in the table efficiently
        query = f'''
        SELECT cfs_run FROM {table}
        ORDER BY cfs_run DESC
        LIMIT 1;
        '''
        cursor.execute(query)
        result = cursor.fetchone()

        # Close the connection to the database
        conn.close()

        if result:
            # Get the last cfs_run
            last_cfs_run = result[0]

            # Convert cfs_run (integer) to datetime object (format YYYYMMDDHH)
            last_datetime = datetime.strptime(str(last_cfs_run), '%Y%m%d%H')

            # Determine the next forecast time based on the last cfs_run
            if last_datetime.hour == 18:
                # If the last hour was 18, increment to the next day at 00:00
                next_datetime = last_datetime + timedelta(days=1)
                next_hour = 0
            elif last_datetime.hour == 12:
                next_hour = 18
                next_datetime = last_datetime
            elif last_datetime.hour == 6:
                next_hour = 12
                next_datetime = last_datetime
            else:
                # If the last hour was 0 or any other, increment to 06:00
                next_hour = 6
                next_datetime = last_datetime

            # Format the next CFS run as MM-DD-YYYY HH
            next_cfs_run = next_datetime.strftime('%m-%d-%Y') + " " + str(next_hour).zfill(2)
            return next_cfs_run

        else:
            print("ERROR: No CFS run entries found in the database. Ensure the table contains data.")
            return None

    except sqlite3.Error as e:
        print(f"ERROR accessing the database: {e}")
        return None
    
