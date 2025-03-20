import urllib.request
from bs4 import BeautifulSoup
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
import sqlite3
from datetime import datetime, timedelta

def download_grb2_ncei(product, url_path, download_dir):

    # File counter
    num_files_downloaded = 0

    try:
        response = urllib.request.urlopen(url_path)
        html_content = response.read().decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')
        links = soup.find_all('a', href=lambda href: href and href.startswith(product) and href.endswith('grib.grb2'))
        
        for link in links:
            file_url = url_path + link['href']
            filename = link['href'].split('/')[-1]
            file_path = os.path.join(download_dir, filename)
            urllib.request.urlretrieve(file_url, file_path)
            print(f"Downloaded: {filename}")

            num_files_downloaded += 1

    except Exception as e:
        print(f"An error occurred: {e}")

def download_grb2_aws(product, bucket_name, url_path, download_dir):
    """
    Download the CFS forecast from AWS

    Parameters:
    - product: 'flx' or 'pgb'
    - bucket_name: for CFS data it is 'noaa-cfs-pds'
    - url_path: the url path to data
    - download_dir: location to download data to
    """
    num_files_downloaded = 0

    # Create a boto3 client for S3
    s3_config = Config(signature_version=UNSIGNED)
    s3 = boto3.client('s3', config=s3_config)

    # List all objects in the specified folder path
    continuation_token = None
    objects = []

    # Use a loop to handle pagination
    while True:
        list_objects_args = {'Bucket': bucket_name, 'Prefix': url_path}
        if continuation_token:
            list_objects_args['ContinuationToken'] = continuation_token

        list_objects_response = s3.list_objects_v2(**list_objects_args)

        objects.extend(list_objects_response.get('Contents', []))

        if not list_objects_response.get('IsTruncated', False):
            break

        continuation_token = list_objects_response.get('NextContinuationToken')

    # Iterate over each object and download if it ends with '.grb2'
    for obj in objects:
        key = obj['Key']
        if product in key and key.endswith('grib.grb2'): #if key.endswith('.grb2'):
            local_file_path = os.path.join(download_dir, os.path.relpath(key, url_path))

            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file
            s3.download_file(bucket_name, key, local_file_path)
            num_files_downloaded += 1

            print(f"Downloaded: {key}")

def check_url_exists(url):
    try:
        response = requests.head(url)
        # Check if the response is OK (status code 200)
        return response.status_code == 200
    except requests.RequestException:
        return False
    

def open_database(database):
    # Check if the database file exists
    if not os.path.exists(database):
        print(f"Creating new database: '{database}'.")
    
    # Connect to the database (it will create it if it doesn't exist)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Create table with necessary columns: cfs_run, year, month, lake, surface_type, cnbs, value
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

    conn.commit()
    conn.close()


def pull_from_db(database, table, cfs_run, year, month, lake, surface_type, cnbs):
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Query the database for the corresponding value
    query = f'''
    SELECT value FROM {table}
    WHERE cfs_run = ? AND year = ? AND month = ? AND lake = ? AND surface_type = ? AND cnbs = ?
    '''
    
    cursor.execute(query, (cfs_run, year, month, lake, surface_type, cnbs))

    # Fetch the result (returns None if not found)
    result = cursor.fetchone()
    conn.close()

    # If result is found, return the value, otherwise return None
    if result:
        return result[0]
    else:
        return None
    
def get_next_cfs_run(database, table):
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # Query to get the last entry in the table directly and efficiently
    query = f'''
    SELECT cfs_run FROM {table}
    ORDER BY cfs_run DESC
    LIMIT 1;
    '''
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()

    if result:
        last_cfs_run = result[0]
        
        # Convert the cfs_run (integer) directly to a datetime object
        last_datetime = datetime.strptime(str(last_cfs_run), '%Y%m%d%H')

        # Determine the next time slot (increment hour)
        next_datetime = last_datetime

        if last_datetime.hour == 18:
            next_datetime += timedelta(days=1)
            next_hour = 0
        elif last_datetime.hour == 12:
            next_hour = 18
        elif last_datetime.hour == 6:
            next_hour = 12
        else:
            next_hour = 6
        
        # Create next cfs_run as a string in MM-DD-YYYY HH format
        next_cfs_run = next_datetime.strftime('%m-%d-%Y') + " " + str(next_hour).zfill(2)
        return next_cfs_run
    else:
        print("The last cfs run entry could not be read. Check the database and try again.")
        return None
    
