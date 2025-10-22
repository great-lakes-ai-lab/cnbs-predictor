# To test use "pytest tests/unit/test_data_inputs.py"

import requests
import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
import os
import sys
sys.path.append(os.path.abspath('../../'))
from src.data_downloader import CFSDownloader

class TestURLAvailability:

    # Test to check AWS and NCEI URL are accessible
    def test_url_availability(self):
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