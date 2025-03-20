import os
import requests
import shutil
import glob

def create_directory(directory):
    """Create a directory if it doesn't already exist."""
    try:
        # Check if the directory already exists
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_files(directory, affix, identifier):
    """
    Get a list of all GRIB2 files in the specified directory.

    Parameters:
    - directory (str): Path to the directory containing files.
    - affix (str): 'prefix' or 'suffix'
    - identifier (str):  (ie. 'pgb', 'flx', '.grb2', or '.nc')
    Returns:
    - List of file paths to the GRIB2 files.
    """
    files = []
    for file_name in os.listdir(directory):
        if affix == 'suffix': # ends with
            if file_name.endswith(identifier):
                file_path = os.path.join(directory, file_name)
                files.append(file_path)
        elif affix == 'prefix': # begins with
            if file_name.startswith(identifier):
                file_path = os.path.join(directory, file_name)
                files.append(file_path)
    #sorted_files = sorted(files)
    return files

def check_url_exists(url):
    try:
        response = requests.head(url)
        # Check if the response is OK (status code 200)
        return response.status_code == 200
    except requests.RequestException:
        return False