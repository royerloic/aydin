from pprint import pprint

import numpy
import pyopencl
from pyopencl.array import Array

from pitl.features.fast.features_1d import collect_feature_1d
from pitl.features.fast.features_2d import collect_feature_2d
from pitl.features.fast.features_3d import collect_feature_3d
from pitl.features.fast.features_4d import collect_feature_4d
from pitl.opencl.opencl_provider import OpenCLProvider


def test_collect_feature_1d():
    n = 13

    opencl_provider = OpenCLProvider()

    # def collect_feature_1d(opencl_provider, image_gpu, integral_image_gpu, feature_gpu, dx, lx, exclude_center=True):

    image_o = numpy.random.rand(n).astype(numpy.float32)
    image_i = image_o.cumsum()

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = pyopencl.array.to_device(opencl_provider.queue, image_i)
    feature_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    collect_feature_1d(
        opencl_provider,
        image_o_gpu,
        image_i_gpu,
        feature_gpu,
        0,
        1,
        exclude_center=False,
        optimisation=False,
    )

    is_equal = numpy.abs(feature_gpu.get() - image_o) < 0.001

    # pprint(is_equal)
    # pprint(image_o)
    # pprint(feature_gpu.get())

    assert numpy.all(is_equal)


def test_collect_feature_2d():
    w = 3
    h = 7

    opencl_provider = OpenCLProvider()

    # def collect_feature_2d(opencl_provider, image_gpu, integral_image_gpu, feature_gpu, dx, lx, exclude_center=True):

    image_o = numpy.random.rand(h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = pyopencl.array.to_device(opencl_provider.queue, image_i)
    feature_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    collect_feature_2d(
        opencl_provider,
        image_o_gpu,
        image_i_gpu,
        feature_gpu,
        0,
        0,
        1,
        1,
        exclude_center=False,
        optimisation=False,
    )

    is_equal = numpy.abs(feature_gpu.get() - image_o) < 0.001

    # pprint(is_equal)
    # pprint(image_o)
    # pprint(feature_gpu.get())

    assert numpy.all(is_equal)


def test_collect_feature_3d():
    w = 3
    h = 7
    d = 5

    opencl_provider = OpenCLProvider()

    # def collect_feature_2d(opencl_provider, image_gpu, integral_image_gpu, feature_gpu, dx, lx, exclude_center=True):

    image_o = numpy.random.rand(d, h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = pyopencl.array.to_device(opencl_provider.queue, image_i)
    feature_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    collect_feature_3d(
        opencl_provider,
        image_o_gpu,
        image_i_gpu,
        feature_gpu,
        0,
        0,
        0,
        1,
        1,
        1,
        exclude_center=False,
        optimisation=False,
    )

    is_equal = numpy.abs(feature_gpu.get() - image_o) < 0.001

    pprint(is_equal)
    pprint(image_o)
    pprint(feature_gpu.get())

    assert numpy.all(is_equal)


def test_collect_feature_4d():
    w = 3
    h = 7
    d = 5
    e = 2

    opencl_provider = OpenCLProvider()

    # def collect_feature_2d(opencl_provider, image_gpu, integral_image_gpu, feature_gpu, dx, lx, exclude_center=True):

    image_o = numpy.random.rand(e, d, h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=3).cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = pyopencl.array.to_device(opencl_provider.queue, image_i)
    feature_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    collect_feature_4d(
        opencl_provider,
        image_o_gpu,
        image_i_gpu,
        feature_gpu,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        exclude_center=False,
        optimisation=False,
    )
    is_equal = numpy.abs(feature_gpu.get() - image_o) < 0.001

    pprint(is_equal)
    pprint(image_o)
    pprint(feature_gpu.get())

    assert numpy.all(is_equal)
