import pytest
from os import path

import numpy
from numpy import percentile

from aydin.io import imread
from aydin.io.datasets import examples_single
from aydin.normaliser.minmax import MinMaxNormaliser
from aydin.normaliser.percentile import PercentileNormaliser


def test_percentile_normaliser():

    input_path = examples_single.pourquie_elec.get_path()

    assert path.exists(input_path)

    _test_percentile_normaliser_internal(input_path, False)


def _test_percentile_normaliser_internal(input_path, use_dask):

    array, metadata = imread(input_path, zarr_cache=use_dask)
    print(array.shape)

    assert array.dtype == numpy.uint8

    percent = 0.001
    normaliser = PercentileNormaliser(percent=percent)
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    assert 0.0 <= normalised_array.min() and normalised_array.max() <= 1.0

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint8

    rmin = percentile(denormalised_array, 100 * percent)
    rmax = percentile(denormalised_array, 100 - 100 * percent)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 20


def test_minmax_normaliser():

    input_path = examples_single.pourquie_elec.get_path()

    assert path.exists(input_path)

    _test_minmax_normaliser_internal(input_path, False)


def _test_minmax_normaliser_internal(input_path, use_dask):

    array, metadata = imread(input_path, zarr_cache=use_dask)
    array = array[0]
    print(array.shape)

    assert array.dtype == numpy.uint8

    normaliser = MinMaxNormaliser()
    normaliser.calibrate(array)
    print(f"before normalisation: min,max = {(normaliser.rmin, normaliser.rmax)}")
    new_array = array.copy()
    normalised_array = normaliser.normalise(new_array)

    assert normalised_array.dtype == numpy.float32

    # normalised_array *= 2
    denormalised_array = normaliser.denormalise(normalised_array)

    assert denormalised_array.dtype == numpy.uint8

    rmin = numpy.min(denormalised_array)
    rmax = numpy.max(denormalised_array)
    print(f"after normalisation: min,max = {(rmin, rmax)}")

    assert abs(normaliser.rmin - rmin) < 5 and abs(normaliser.rmax - rmax) < 5
