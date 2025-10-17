# To test use "pytest tests/unit/test_data_inputs.py"

import requests
import boto3
import pytest
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

def test_url_availability():
    print("Running NCEI availability test...")
    urls = [
        "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast/monthly-means/",
        "https://noaa-cfs-pds.s3.amazonaws.com/"
    ]
    for url in urls:
        response = requests.head(url, timeout=10)
        assert response.status_code == 200, f"Endpoint not reachable: {url}"

def test_aws_bucket_accessibility():
    """
    Simple test to ensure AWS S3 is reachable and the target bucket exists.
    """
    bucket_name = "noaa-cfs-pds"  # Replace with your expected bucket

    # Create an unsigned (public) S3 client
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    
    try:
        # Head bucket is a lightweight way to confirm it exists and is reachable
        s3.head_bucket(Bucket=bucket_name)
        print(f"✅ AWS S3 bucket '{bucket_name}' is available.")
    except EndpointConnectionError:
        pytest.fail("AWS S3 endpoint not reachable — check network or credentials.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            pytest.fail(f"Bucket '{bucket_name}' does not exist or has changed.")
        else:
            pytest.fail(f"AWS S3 error occurred: {e}")

