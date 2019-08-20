import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from aydin.io.folders import get_home_folder

home_path = get_home_folder()
aydin_path = home_path + "/.aydin"
if not os.path.exists(aydin_path + "/plaidml"):
    os.makedirs(aydin_path + "/plaidml")

if not os.path.exists(aydin_path + "/plaidml/libplaidml.dylib"):
    output = aydin_path + "/plaidml/libplaidml.dylib"
    gdd.download_file_from_google_drive(
        file_id='1VvdBli5leKug2niTitw2ikO3Qj_N-5C6', dest_path=output, unzip=False
    )
if not os.path.exists(aydin_path + "/plaidml/config.json"):
    output = aydin_path + "/plaidml/config.json"
    gdd.download_file_from_google_drive(
        file_id='1j7hqCFKugIc56ti7gk_uCfui6e-OQM7T', dest_path=output, unzip=False
    )
if not os.path.exists(aydin_path + "/plaidml/experimental.json"):
    output = aydin_path + "/plaidml/experimental.json"
    gdd.download_file_from_google_drive(
        file_id='1Y_DOwmpzneGvb91Nt9KnBIlC_zKtIQ9t', dest_path=output, unzip=False
    )

os.environ["PLAIDML_DEFAULT_CONFIG"] = home_path + "/.aydin/plaidml/config.json"
os.environ["PLAIDML_EXPERIMENTAL_CONFIG"] = (
    home_path + "/.aydin/plaidml/experimental.json"
)
