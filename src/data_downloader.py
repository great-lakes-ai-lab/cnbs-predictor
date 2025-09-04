import os
import urllib.request
import requests
import boto3
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import pandas as pd

from src.utilities import check_url_exists

class CFSDownloader:
    """
    A utility class to download Climate Forecast System (CFSv2) GRIB2 forecast data
    from AWS or the NOAA NCEI archive.

    This class manages local storage paths, S3 configurations for public AWS access,
    and sets up the environment for bulk data retrieval operations using threaded downloads.

    Attributes
    ----------
    base_download_dir : str
        Root directory where all data will be downloaded.

    cfs_dir : str
        Subdirectory for storing CFS-specific data under the base download directory.

    s3_config : botocore.config.Config
        Boto3 configuration with unsigned (public) access for AWS S3.

    s3 : boto3.client
        S3 client used to communicate with the NOAA AWS public dataset.
    """

    def __init__(self):
        """
        Initialize the DataDownloader class.

        This method configures an S3 client for public access to download CFS data.
        """
        self.s3_config = Config(signature_version=UNSIGNED)
        self.s3 = boto3.client('s3', config=self.s3_config)


    def download(self, download_dir, start_date, end_date=None, hours=None, products=None, source=None, bucket_name=None):
        """
        Download Climate Forecast System v2 (CFSv2) GRIB2 forecast files from AWS or NCEI over a specified date range.

        This method automates the download of monthly mean CFSv2 GRIB2 forecast files for specified hours, products,
        and source (AWS or NCEI). Files are organized by date and stored in local subdirectories named after each date.

        Parameters
        ----------
        download_dir : str
            Path to the local root directory where all downloaded files will be stored.

        start_date : str
            Start date for the download period, in 'MM-DD-YYYY' format.

        end_date : str, optional
            End date for the download period, in 'MM-DD-YYYY' format.
            If not provided, only start_date will be used.

        hours : list of str, optional
            List of forecast cycle hours to retrieve. Valid options are ['00', '06', '12', '18'].
            If not provided, all hours are downloaded.

        products : list of str, optional
            List of product prefixes to filter downloads. Typical values include:
            - 'pgbf': Pressure-level fields
            - 'flxf': Flux fields
            - 'ocnf': Ocean forecast fields
            - 'ipvf': Intermediate forecast variables
            If not provided, all default products are downloaded.

        source : str, optional
            The data source to use. Must be either:
            - 'aws': Download from NOAA AWS public dataset (default)
            - 'ncei': Download from NOAA's National Centers for Environmental Information

        bucket_name : str, optional
            The name of the AWS S3 bucket. Defaults to 'noaa-cfs-pds' when using AWS.
            Not used when `source='ncei'`.

        Raises
        ------
        ValueError
            If date formats are incorrect, if invalid product names are provided, or if an invalid source is specified.

        Notes
        -----
        - If only start_date is provided, downloads only that day.
        - Otherwise, downloads all dates from start_date (inclusive) to end_date (exclusive).
        - Skips missing files or directories gracefully.
        - AWS downloads use S3 API calls; NCEI uses HTML scraping.
        """

        os.makedirs(download_dir, exist_ok=True)

        # Default values
        if hours is None:
            hours = ['00', '06', '12', '18']
        if products is None:
            products = ['pgbf', 'flxf', 'ocnf', 'ipvf']
        if source is None:
            source = 'aws'
        if bucket_name is None:
            bucket_name = 'noaa-cfs-pds'

        # Validate and parse start_date
        try:
            start = datetime.strptime(start_date, "%m-%d-%Y")
        except ValueError:
            raise ValueError("ERROR: start_date must be in 'MM-DD-YYYY' format.")

        # Handle optional end_date
        if end_date:
            try:
                end = datetime.strptime(end_date, "%m-%d-%Y")
            except ValueError:
                raise ValueError("ERROR: end_date must be in 'MM-DD-YYYY' format.")
            if end < start:
                raise ValueError("ERROR: end_date must be after or equal to start_date")
        else:
            end = start + timedelta(days=1)

        # Validate products
        if not isinstance(products, list) or not all(isinstance(p, str) for p in products):
            raise ValueError("ERROR: Products must be a list of strings.")
        valid_products = {'pgbf', 'flxf', 'ocnf', 'ipvf', 'pgb', 'flx', 'ocn', 'ipv'}

        delta = timedelta(days=1)
        current = start

        while current < end:
            YYYY = current.strftime("%Y")
            MM = current.strftime("%m")
            DD = current.strftime("%d")
            date_folder = current.strftime("%Y%m%d")
            date_dir = os.path.join(download_dir, date_folder)
            os.makedirs(date_dir, exist_ok=True)

            for HH in hours:
                for product in products:
                    if product not in valid_products:
                        print(f"ERROR: No files matching '{product}'. Valid products are: {', '.join(valid_products)}. Skipping.")
                        continue

                    if source.lower() == 'aws':
                        url_path = f'cfs.{YYYY}{MM}{DD}/{HH}/monthly_grib_01/'

                        # Check if any files exist in AWS
                        continuation_token = None
                        objects_found = False
                        while True:
                            list_args = {'Bucket': bucket_name, 'Prefix': url_path}
                            if continuation_token:
                                list_args['ContinuationToken'] = continuation_token

                            response = self.s3.list_objects_v2(**list_args)
                            contents = response.get('Contents', [])
                            if any(product in os.path.basename(obj['Key']) and obj['Key'].endswith('grib.grb2') for obj in contents):
                                objects_found = True
                                break
                            if not response.get('IsTruncated', False):
                                break
                            continuation_token = response.get('NextContinuationToken')

                        if not objects_found:
                            print(f"No AWS files for '{product}' on {MM}-{DD}-{YYYY} {HH}. Skipping.")
                            continue

                        self._from_aws(product, bucket_name, url_path, date_dir)

                    elif source.lower() == 'ncei':
                        base_url = 'https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means'
                        url_path = f'{base_url}/{YYYY}/{YYYY}{MM}/{YYYY}{MM}{DD}/{YYYY}{MM}{DD}{HH}/'

                        if not check_url_exists(url_path):
                            print(f"No NCEI URL for {product} on {MM}-{DD}-{YYYY} {HH}. Skipping.")
                            continue

                        try:
                            response = urllib.request.urlopen(url_path)
                            soup = BeautifulSoup(response.read().decode('utf-8'), 'html.parser')
                            links = [link for link in soup.find_all('a') if link['href'].startswith(product) and link['href'].endswith('grib.grb2')]
                            if not links:
                                print(f"No NCEI files for '{product}' on {MM}-{DD}-{YYYY} {HH}. Skipping.")
                                continue
                        except Exception as e:
                            print(f"Error accessing NCEI URL for {MM}-{DD}-{YYYY} {HH}: {e}")
                            continue

                        self._from_ncei(product, url_path, date_dir)

                    else:
                        raise ValueError('Invalid source. Source must be either "aws" or "ncei".')

            current += delta

        print("Download completed successfully.")


    def _from_aws(self, product, bucket_name, url_path, download_dir):
        """
        Download GRIB2 files from an AWS S3 bucket for a specific product and path.

        Parameters
        ----------
        product : str
            The product prefix to filter files (e.g., 'pgbf', 'flxf').

        bucket_name : str
            The AWS S3 bucket to download from (e.g., 'noaa-cfs-pds').

        url_path : str
            The S3 object prefix (path) within the bucket.

        download_dir : str
            Local directory where the downloaded files will be stored.

        Notes
        -----
        - Only files ending in '.grib.grb2' and matching the product prefix will be downloaded.
        - Uses `ThreadPoolExecutor` to speed up downloads with parallel threads.
        - Prints status updates for each downloaded file.
        """

        continuation_token = None
        objects = []

        while True:
            list_args = {'Bucket': bucket_name, 'Prefix': url_path}
            if continuation_token:
                list_args['ContinuationToken'] = continuation_token

            response = self.s3.list_objects_v2(**list_args)
            contents = response.get('Contents', [])
            if not contents:
                print(f"No files found in AWS path: {url_path}")
                break

            objects.extend(contents)
            if not response.get('IsTruncated', False):
                break
            continuation_token = response.get('NextContinuationToken')

        # Filter keys for products
        keys_to_download = [obj['Key'] for obj in objects if product in os.path.basename(obj['Key']) and obj['Key'].endswith('grib.grb2')]

        def download_key(key):
            filename = os.path.basename(key)
            local_path = os.path.join(download_dir, filename)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(bucket_name, key, local_path)
            print(f"Downloaded from AWS: {filename}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(download_key, keys_to_download)
            
    def _from_ncei(self, product, url_path, download_dir):
        """
        Download GRIB2 files from the NCEI HTTP server for a specified product.

        Parameters
        ----------
        product : str
            The product prefix used to filter downloadable GRIB2 files.

        url_path : str
            Full HTTP URL pointing to the NCEI folder containing GRIB2 files.

        download_dir : str
            Local directory where files will be saved.

        Notes
        -----
        - Parses the HTML directory listing from NCEI using BeautifulSoup.
        - Downloads matching `.grib.grb2` files using `urllib.request.urlretrieve`.
        - Uses threading to speed up downloading.
        - Gracefully handles HTTP and URL errors.
        """

        try:
            response = urllib.request.urlopen(url_path)
            soup = BeautifulSoup(response.read().decode('utf-8'), 'html.parser')
            links = soup.find_all('a', href=lambda href: href and href.startswith(product) and href.endswith('grib.grb2'))

            def download_file(link):
                filename = link['href']
                file_url = url_path + filename
                local_path = os.path.join(download_dir, filename)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                urllib.request.urlretrieve(file_url, local_path)
                print(f"Downloaded from NCEI: {filename}")

            with ThreadPoolExecutor(max_workers=5) as executor:
                executor.map(download_file, links)

        except Exception as e:
            print(f"ERROR downloading from NCEI: {e}")

class SSTDownloader:
    def __init__(self):
        pass

    def current_glsea(self, url, units='K'):
        """
        Fetch the most recent daily lake average surface water temperature
        from a NOAA GLSEA .dat file.

        Parameters
        ----------
        url : str
            The URL of the .dat file.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: Date, Superior, Michigan-Huron, Erie, Ontario
            or None if the URL is not accessible.
        """
        # Check if URL exists before proceeding
        if not check_url_exists(url):
            print(f"URL does not exist or is not reachable: {url}")
            return None

        # Download file as text
        response = requests.get(url)
        response.raise_for_status()
        text_data = response.text

        # Keep only data lines (start with a digit)
        lines = text_data.strip().split("\n")
        data_lines = [line for line in lines if line.strip() and line[0].isdigit()]

        # Get the last line (most recent record)
        last_line = data_lines[-1]

        # Column names
        col_names = ["year", "day", "superior_sst", "michigan_sst", "huron_sst", "erie_sst", "ontario_sst", "stclair_sst"]

        # Read the last line into a DataFrame
        df = pd.read_csv(StringIO(last_line), sep=r"\s+", names=col_names)

        # Convert Year + Julian Day → datetime
        df["Date"] = pd.to_datetime(df["year"].astype(str) + df["day"].astype(str), format="%Y%j")

        # Compute Michigan-Huron average
        df["michigan-huron_sst"] = df[["michigan_sst", "huron_sst"]].mean(axis=1)
        
        # Select and optionally convert units
        df = df.set_index("Date")[["superior_sst", "michigan-huron_sst", "erie_sst", "ontario_sst"]]

        if units.upper() == "K":
            df = df + 273.15
        elif units.upper() != "C":
            raise ValueError("Unsupported units. Use 'C' for Celsius or 'K' for Kelvin.")

        return df

    def all_glsea(self, url, units='K'):
        """
        Fetch all daily lake surface water temperatures from a NOAA GLSEA .dat file URL.

        Parameters
        ----------
        url : str
            The URL of the .dat file (GLSEA daily data).
        units : str, optional
            Unit system for output. 'C' for Celsius, 'K' for Kelvin. Default is 'K'.

        Returns
        -------
        pd.DataFrame
            A DataFrame with Date, Superior, Michigan-Huron, Erie, Ontario columns.
            Returns None if the URL is not accessible.
        """
        if not check_url_exists(url):
            print(f"URL does not exist or is not reachable: {url}")
            return None

        # Download full file
        response = requests.get(url)
        response.raise_for_status()
        text_data = response.text

        # Filter lines that start with a digit (data lines)
        lines = text_data.strip().split("\n")
        data_lines = [line for line in lines if line.strip() and line[0].isdigit()]
        full_data = "\n".join(data_lines)

        # Column names from NOAA format
        col_names = ["year", "day", "superior_sst", "michigan_sst", "huron_sst", "erie_sst", "ontario_sst", "stclair_sst"]

        # Read all rows into DataFrame
        df = pd.read_csv(StringIO(full_data), sep=r"\s+", names=col_names)

        # Date conversion from year + Julian day
        df["Date"] = pd.to_datetime(df["year"].astype(str) + df["day"].astype(str), format="%Y%j")

        # Michigan-Huron average
        df["michigan-huron_sst"] = df[["michigan_sst", "huron_sst"]].mean(axis=1)

        # Select and optionally convert units
        df = df.set_index("Date")[["superior_sst", "michigan-huron_sst", "erie_sst", "ontario_sst"]]

        if units.upper() == "K":
            df = df + 273.15
        elif units.upper() != "C":
            raise ValueError("Unsupported units. Use 'C' for Celsius or 'K' for Kelvin.")

        return df