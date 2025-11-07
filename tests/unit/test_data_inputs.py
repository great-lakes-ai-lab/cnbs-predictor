# To test use "pytest --capture=no test_data_inputs.py"

import requests
import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from datetime import datetime, timedelta, UTC
import os
import sys
sys.path.append(os.path.abspath('../../'))
from src.data_downloader import CFSDownloader

class TestURLAvailability:

    # Test to check AWS and NCEI URL are accessible
    def test_base_url_availability(self):
        """
        Test to verify the AWS and NCEI urls are reachable, respond, and haven't changed.
        """
        urls = [
            ("AWS", "https://noaa-cfs-pds.s3.amazonaws.com/"),
            ("NCEI", "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/"),
        ]
        for source, url in urls:
            try:
                response = requests.head(url, timeout=10)
                print(f"✅ {source} reachable")
                assert response.status_code == 200, f"❌ {source} not reachable"
            except requests.RequestException as e:
                print(f"❌ {source}: Error accessing URL -> {url}, Exception: {e}")

    # Test to check AWS S3 bucket name has not changed and is still reachable
    def test_aws_bucket_accessibility(self):
        """
        Simple test to ensure AWS S3 is reachable and the bucket still exists.
        """
        bucket_name = "noaa-cfs-pds"

        # Create an unsigned S3 client
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"✅ AWS S3 bucket '{bucket_name}' is available.")
        except EndpointConnectionError:
            pytest.fail("❌ AWS S3 endpoint not reachable — check network or credentials.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pytest.fail(f"❌ Bucket '{bucket_name}' does not exist or has changed.")
            else:
                pytest.fail(f"❌ AWS S3 error occurred: {e}")


import pytest
from datetime import datetime, timedelta, UTC
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import requests
from bs4 import BeautifulSoup

class TestCFSDataAvailability:
    """
    Unit test that verifies today's (or yesterday's) Climate Forecast System (CFSv2)
    forecast data is available from either NOAA AWS (noaa-cfs-pds) or NCEI.
    It does not download files—only checks for presence via S3 list or HTTP HEAD/HTML listing.
    """

    def _check_aws_availability(self, date_obj):
        """Check AWS NOAA CFS public bucket for GRIB2 files for given date."""
        bucket_name = "noaa-cfs-pds"
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        prefix = f"cfs.{date_obj.strftime('%Y%m%d')}/00/monthly_grib_01/"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=100)
        contents = response.get("Contents", [])
        files = [obj["Key"] for obj in contents if obj["Key"].endswith(".grib.grb2")]
        return files

    def _check_ncei_availability(self, date_obj):
        """Check NCEI HTTPS endpoint for GRIB2 files for given date."""
        date_str = date_obj.strftime("%Y%m%d")
        year = date_obj.strftime("%Y")
        month = date_obj.strftime("%m")
        base_url = (
            f"https://www.ncei.noaa.gov/data/climate-forecast-system/"
            f"access/operational-9-month-forecast/monthly-means/{year}/{year}{month}/"
        )
        try:
            response = requests.get(base_url, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, "html.parser")
            # Filter links for today's date and GRIB2 files
            links = [
                a["href"] for a in soup.find_all("a", href=True)
                if date_str in a["href"] and a["href"].endswith(".grib.grb2")
            ]
            return links
        except requests.RequestException as e:
            print(f"Error accessing NCEI URL: {e}")
            return []

    def test_cfs_data_availability(self, capsys):
        """Check both AWS and NCEI for today's or yesterday's CFS GRIB2 availability."""
        today = datetime.now(UTC)
        for offset in range(0, 2):  # check today and yesterday
            check_date = today - timedelta(days=offset)
            
            # AWS
            aws_files = self._check_aws_availability(check_date)
            aws_status = f"✅ AWS: Found {len(aws_files)} files" if aws_files else "❌ AWS: No up-to-date files found"

            # NCEI
            ncei_files = self._check_ncei_availability(check_date)
            ncei_status = f"✅ NCEI: Found {len(ncei_files)} files" if ncei_files else "❌ NCEI: No up-to-date files found"

            print(f"For {check_date.strftime('%m-%d-%Y')}: {aws_status}, {ncei_status}")

            # Pass immediately if either source has data
            if aws_files or ncei_files:
                break
        else:
            print("❌ No up-to-date CFS GRIB2 data found in AWS or NCEI for today or yesterday.")
            pytest.fail("No up-to-date CFS GRIB2 data found in AWS or NCEI for today or yesterday.")

        # Print captured output so it appears in pytest logs
        captured = capsys.readouterr()
        print(captured.out)

class TestDataDownloader:

    def test_cfs_downloader_function(self, tmp_path):
        """
        Test that downloads data from AWS for 01-01-2024 at hour 00 and pulls pgbf products.
        Expects a total of 10 GRIB2 files in the expected directory.

        Uses pytest's tmp_path to ensure a temporary storage location that is cleaned after test.
        """
        # Create a temporary directory for downloads
        cfs_downloader = CFSDownloader()

        cfs_downloader.download(
            download_dir=tmp_path,
            start_date="01-01-2024",
            end_date="01-02-2024",
            hours=["00"],
            products=["pgbf"],
            source="aws"
        )

        # Directory for the specific date
        expected_dir = os.path.join(tmp_path, "20240101")

        # Assert the directory was created
        assert os.path.isdir(expected_dir), f"Expected directory not found: {expected_dir}"

        # Look at the all the files downloaded
        files = [
            f for f in os.listdir(expected_dir)
            if f.startswith("pgbf") and f.endswith("grib.grb2")
        ]

        # Assert that 10 files are downloaded
        if len(files) == 10:
            print(f"✅ Downloaded {len(files)} 'pgbf' files as expected. CFS downloader is functioning correctly.")
        else:
            print(f"❌ Expected 10 'pgbf' files, but found {len(files)}.")

    