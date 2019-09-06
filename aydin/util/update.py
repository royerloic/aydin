import json
import os
import platform
import requests

from google_drive_downloader import GoogleDriveDownloader as gdd


# Return version and id of latest version
def get_latest_version_details():
    r = requests.get("https://api.github.com/gists/b99bfed533738200a82dbf22d5406a9e")
    versions_json = json.loads(r.json()['files']['secret.json']['content'])
    latest_version = list(versions_json[str(platform.system())].keys())[-1]
    latest_id = versions_json[str(platform.system())][latest_version]

    return latest_version, latest_id


# Download specific version of executable
def download_specific_version(file_version, file_id):
    # Generate file name
    name = "aydin_" + file_version

    # Add prefix for windows host
    if platform.system() == 'Windows':
        name += ".exe"

    # Download the latest version
    updated_app_path = os.path.join(str(os.getcwd()), name)
    gdd.download_file_from_google_drive(
        file_id=file_id, dest_path=updated_app_path, unzip=False
    )
    print("Please find the most recent version here: ", updated_app_path)

    # Change permissions for new downloaded executable if needed
    if platform.system() == 'Darwin':
        os.system("chmod 755 " + updated_app_path)

    # Return path for downloaded version
    return updated_app_path
