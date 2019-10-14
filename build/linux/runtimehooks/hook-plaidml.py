import gdown
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

from aydin.io.folders import get_home_folder

home_path = get_home_folder()
aydin_path = home_path + "/.aydin"
if not os.path.exists(aydin_path + "/plaidml"):
    os.makedirs(aydin_path + "/plaidml")

if not os.path.exists(aydin_path + "/plaidml/libplaidml.so"):
    output = aydin_path + "/plaidml/libplaidml.so"

    os.system("wget https://gitlab.com/AhmetCanSolak/linuxbinaryplaidml/raw/master/libplaidml.so?inline=false -O " + output)
#     # url = "https://drive.google.com/file/d/1F6gCJQMGAda9udd67ESohVhBMLWoJPHr/view"
#
#     # gdown.download(url, output, quiet=False)
#     gdd.download_file_from_google_drive(
#         file_id='1NtWjYoUe4m1VD9lcoAUNyvcq2B22TZyG', dest_path=output, unzip=False
#     )
#
#     os.system("ls " + aydin_path + "/plaidml/")
#     os.system("mv " + output + " " + aydin_path + "/plaidml/libplaidml.so")
#     os.system("ls " + aydin_path + "/plaidml/")
    #
    # import zipfile
    # with zipfile.ZipFile(output, 'r') as zip_ref:
    #     zip_ref.extractall(aydin_path + "/plaidml/")

    # outputa = aydin_path + "/plaidml/alibplaidml.so"
    # gdd.download_file_from_google_drive(
    #     file_id='1Xsx7moYUDlTNbJVXSFTBjcsSNLKNl7dH', dest_path=output, unzip=False
    # )
    # os.system("cp " + outputa + " " + output)

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
