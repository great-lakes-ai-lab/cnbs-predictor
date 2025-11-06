# To test use "pytest tests/unit/test_data_inputs.py"

import requests
import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.abspath('../../'))
from src.data_downloader import CFSDownloader

class TestURLAvailability:

    # Test to check AWS and NCEI URL are accessible
    def test_base_url_availability(self):
        """
        Test to verify the NCEI and AWS urls are reachable, respond, and haven't changed.
        """
        urls = [
            "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/",
            "https://noaa-cfs-pds.s3.amazonaws.com/"
        ]
        for url in urls:
            response = requests.head(url, timeout=10)
            assert response.status_code == 200, f"Endpoint not reachable: {url}"

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
            pytest.fail("AWS S3 endpoint not reachable — check network or credentials.")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                pytest.fail(f"Bucket '{bucket_name}' does not exist or has changed.")
            else:
                pytest.fail(f"AWS S3 error occurred: {e}")


class TestCFSDataAvailability:
    """
    Unit test that verifies today's (or yesterday's) Climate Forecast System (CFSv2)
    forecast data is available from either NOAA AWS (noaa-cfs-pds) or NCEI.
    It does not download files—only checks for presence via S3 list or HTTP HEAD.
    """

    def _check_aws_availability(self, date_obj):
        """Check AWS NOAA CFS public bucket for GRIB2 files for given date."""
        bucket_name = "noaa-cfs-pds"
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        prefix = f"cfs.{date_obj.strftime('%Y%m%d')}/00/monthly_grib_01/"
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=10)
        contents = response.get("Contents", [])
        return [obj["Key"] for obj in contents if obj["Key"].endswith(".grib.grb2")]

    def _check_ncei_availability(self, date_obj):
        """Check NCEI HTTPS endpoint for GRIB2 availability using a HEAD request."""
        # NCEI CFS data base path example
        # https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/2024/202401/pgbf.01.2024010100.202402.avrg.grib.grb2
        date_str = date_obj.strftime("%Y%m%d")
        year = date_obj.strftime("%Y")
        example_url = (
            f"https://www.ncei.noaa.gov/data/climate-forecast-system/"
            f"access/operational-9-month-forecast/monthly-means/{year}/{date_str[:6]}/"
            f"pgbf.01.{date_str}00.202402.avrg.grib.grb2"
        )

        try:
            response = requests.head(example_url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def test_cfs_data_availability(self):
        """Check for AWS or NCEI CFSv2 GRIB2 availability (today or yesterday)."""
        today = datetime.utcnow()
        for offset in range(0, 2):  # Check today and yesterday
            check_date = today - timedelta(days=offset)

            # AWS check
            aws_files = self._check_aws_availability(check_date)
            if aws_files:
                print(f"✅ AWS: Found {len(aws_files)} files for {check_date.strftime('%Y-%m-%d')} (00 UTC)")
                return
            else:
                print(f"ℹ️ AWS: No files found for {check_date.strftime('%Y-%m-%d')} (00 UTC)")

            # NCEI check
            if self._check_ncei_availability(check_date):
                print(f"✅ NCEI: Found GRIB2 data for {check_date.strftime('%Y-%m-%d')} (00 UTC)")
                return
            else:
                print(f"ℹ️ NCEI: No data found for {check_date.strftime('%Y-%m-%d')} (00 UTC)")

        pytest.fail("❌ No up-to-date CFS GRIB2 data found in AWS or NCEI for today or yesterday.")

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
        assert len(files) == 10, f"Expected 10 'pgbf' files, but found {len(files)}"

    