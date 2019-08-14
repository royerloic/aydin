import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from aydin.io.folders import get_home_folder

home_path = get_home_folder()
aydin_path = home_path + r"\.aydin"
if not os.path.exists(aydin_path + r"\plaidml"):
    os.makedirs(aydin_path + r"\plaidml")

if not os.path.exists(aydin_path + r"\plaidml\plaidml.dll"):
    output = aydin_path + r"\plaidml\plaidml.dll"
    gdd.download_file_from_google_drive(
        file_id='1ycYVIhzhqNcIyxSBS_N-asW7-aeQ7Yp5', dest_path=output, unzip=False
    )
if not os.path.exists(aydin_path + r"\plaidml\config.json"):
    output = aydin_path + r"\plaidml\config.json"
    gdd.download_file_from_google_drive(
        file_id='1j7hqCFKugIc56ti7gk_uCfui6e-OQM7T', dest_path=output, unzip=False
    )
if not os.path.exists(aydin_path + r"\plaidml\experimental.json"):
    output = aydin_path + r"\plaidml\experimental.json"
    gdd.download_file_from_google_drive(
        file_id='1Y_DOwmpzneGvb91Nt9KnBIlC_zKtIQ9t', dest_path=output, unzip=False
    )

os.environ["PLAIDML_DEFAULT_CONFIG"] = home_path + r"\.aydin\plaidml\config.json"
os.environ["PLAIDML_EXPERIMENTAL_CONFIG"] = (
    home_path + r"\.aydin\plaidml\experimental.json"
)
