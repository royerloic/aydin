import pytest
from os import path
from os.path import join

import dask
import numpy

from aydin.io import io
from aydin.io.datasets import examples_single, examples_zipped
from aydin.io.folders import get_temp_folder
from aydin.io.io import convert_to_zarr, cache_as_zarr, imread, imwrite


def test_analysis():

    for example in examples_single:

        example_file_path = example.get_path()

        analysis_result = io.analyse(example_file_path)

        print(analysis_result)


def test_convert_to_zarr():

    input_path = examples_single.gardner_org.get_path()
    zarr_path = join(get_temp_folder(), 'test_convert2zarr')

    print(zarr_path)

    convert_to_zarr(input_path, zarr_path, overwrite=True)

    assert path.exists(zarr_path)


def test_cache_as_zarr():

    input_path = examples_single.gardner_org.get_path()

    zarr_path = cache_as_zarr(input_path)

    print(zarr_path)

    assert path.exists(zarr_path)

    array, metadata = imread(zarr_path)

    print(array.shape)
    print(metadata)

    assert array.shape == (1, 1, 1, 3, 320, 865, 1014, 1)


def test_opening_with_cache():

    input_path = examples_single.gardner_org.get_path()

    assert path.exists(input_path)

    array, metadata = imread(input_path, zarr_cache=True)

    print(array.shape)

    assert array.shape == (1, 1, 1, 3, 320, 865, 1014, 1)

    # Can we compute the robust min and max ?
    rmin = dask.array.percentile(array.flatten(), 1).compute()
    rmax = dask.array.percentile(array.flatten(), 99).compute()

    print(f"min,max = {(rmin,rmax)}")


@pytest.mark.heavy
def test_opening_tif_folder():

    path = examples_zipped.unser_celegans.get_path()

    array, metadata = imread(path)

    print(array.shape)
    print(metadata)

    assert array.shape == (104, 712, 672)
    assert array.dtype == numpy.uint16


def test_opening_examples():

    for example in examples_single:

        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        if not array is None:
            print(
                f"dataset: {example.value[1]}, shape:{array.shape}, dtype:{array.dtype} "
            )
        else:
            print(f"Cannot open dataset: {example[1]}")


def test_imwrite():

    input_path = examples_single.janelia_flybrain.get_path()

    assert path.exists(input_path)

    array, metadata = imread(input_path, zarr_cache=False)

    temp_file = join(get_temp_folder(), "test_imwrite.tif")

    # We use a generator that takes care of everything:
    with imwrite(temp_file, shape=array.shape, dtype=array.dtype) as tiff_array:

        # We write here the file
        print("Writting file...")
        tiff_array[...] = array

    print("Done...")

    print("Reading back...")
    array2, _ = imread(temp_file)

    assert numpy.all(array == array2)
