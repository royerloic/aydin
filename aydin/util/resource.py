import sys
from urllib.request import urlretrieve
from pathlib import Path
import os
import zipfile

from skimage.exposure import rescale_intensity

import numpy as np

from aydin.io import imread


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS

        return os.path.join(base_path, os.path.basename(relative_path))
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


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
    image, metadata = imread(path)
    bit_depth = int("".join(x for x in image.dtype.name if x.isdigit()))
    if bit_depth <= 16:
        image = image.astype(np.float16)
    else:
        image = image.astype(np.float32)
    return rescale_intensity(image, in_range='image', out_range=(0, 1)), metadata
