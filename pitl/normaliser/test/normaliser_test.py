from os import path

import numpy
from numpy import percentile

from pitl.io import imread
from pitl.io.datasets import examples_single
from pitl.normaliser.minmax import MinMaxNormaliser
from pitl.normaliser.percentile import PercentileNormaliser


def test_percentile_normaliser():

    input_path = examples_single.hyman_hela.get_path()

    assert path.exists(input_path)

    _test_percentile_normaliser_internal(input_path, False)
    _test_percentile_normaliser_internal(input_path, True)


def _test_percentile_normaliser_internal(input_path, use_dask):

    array, metadata = imread(input_path, zarr_cache=use_dask)
    print(array.shape)

    assert array.dtype == numpy.uint16

    percent = 1
    normaliser = PercentileNormaliser(percent=percent)
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint16

    rmin = percentile(denormalised_array, percent)
    rmax = percentile(denormalised_array, 100 - percent)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 5


def test_minmax_normaliser():

    input_path = examples_single.hyman_hela.get_path()

    assert path.exists(input_path)

    _test_minmax_normaliser_internal(input_path, False)
    _test_minmax_normaliser_internal(input_path, True)


def _test_minmax_normaliser_internal(input_path, use_dask):

    array, metadata = imread(input_path, zarr_cache=use_dask)
    array = array[0]
    print(array.shape)

    assert array.dtype == numpy.uint16

    normaliser = MinMaxNormaliser()
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint16

    rmin = numpy.min(denormalised_array)
    rmax = numpy.max(denormalised_array)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 5


# _test_percentile_normaliser()
# _test_minmax_normaliser()
