import pytest
import numpy as np
from tifffile import imread

from ..resource import download_and_extract_zipresource


def test_download_and_extract_zipresource():

    # Testing execution of the method
    try:
        download_and_extract_zipresource(
            url='http://csbdeep.bioimagecomputing.com/example_data/tribolium.zip',
            targetdir='data',
        )
    except Exception:
        pytest.fail("download_and_extract_zipresource failed", True)

    # Read some files
    x = imread('data/tribolium/train/GT/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(
        np.float32
    )
    y = imread('data/tribolium/train/low/nGFP_0.1_0.2_0.5_20_13_late.tif').astype(
        np.float32
    )
    z = imread('data/tribolium/test/GT/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(
        np.float32
    )
    t = imread('data/tribolium/test/low/nGFP_0.1_0.2_0.5_20_14_late.tif').astype(
        np.float32
    )

    # Check if dimensions of images match from different files, we know they should be same
    assert y.shape == x.shape
    assert z.shape == t.shape
