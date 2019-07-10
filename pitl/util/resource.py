from urllib.request import urlretrieve
from pathlib import Path
import os
import zipfile

from skimage.exposure import rescale_intensity

import numpy as np

from pitl.io import imread


def download_and_extract_zipresource(url, targetdir='.'):

    # Check if target directory exists, if not create
    targetdir = Path(targetdir)
    if not targetdir.is_dir():
        targetdir.mkdir(parents=True, exist_ok=True)

    # Compute relative path to resource
    relative_path_to_zip = str(targetdir) + '/' + os.path.basename(url)
    print("relativepath= ", relative_path_to_zip)

    # Check if target resource already exists, retrieve the resource if not exists
    if os.path.exists(relative_path_to_zip[:-4]):
        print("Resource already exists, nothing to download")
    else:
        urlretrieve(url, relative_path_to_zip)
        # Extract the content
        with zipfile.ZipFile(relative_path_to_zip, "r") as zip_ref:
            zip_ref.extractall(str(targetdir))

        # Delete zip file
        Path.unlink(Path(relative_path_to_zip))


def read_image_from_path(path):
    image = imread(path)[0].astype(np.float32)
    return rescale_intensity(image, in_range='image', out_range=(0, 1))
