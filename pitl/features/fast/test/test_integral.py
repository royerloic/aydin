import numpy
import pyopencl
from pyopencl.array import Array

from pitl.features.fast.integral import (
    integral_1d,
    integral_2d,
    integral_3d,
    integral_4d,
)
from pitl.opencl.opencl_provider import OpenCLProvider


def test_integral_1d():
    n = 137

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(n).astype(numpy.float32)
    image_i = image_o.cumsum()

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu = integral_1d(opencl_provider, image_o_gpu, image_i_gpu)
    image_i_r = image_i_gpu.get()

    equality = (
        2 * numpy.abs(image_i_r - image_i) / numpy.abs(image_i_r + image_i) < 0.000001
    )

    for index, is_equal in numpy.ndenumerate(equality):
        if not is_equal:
            print(f"index: {index} value: {is_equal}")
            print(f"image_i = {image_i[index]}, image_i_r = {image_i_r[index]}  ")

    assert numpy.all(equality)


def test_integral_2d():
    w = 31
    h = 71

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu = integral_2d(opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu)
    image_i_r = image_i_gpu.get()

    # pprint(image_o)
    # pprint(image_i)
    # pprint(image_i_r)

    equality = (
        2 * numpy.abs(image_i_r - image_i) / numpy.abs(image_i_r + image_i) < 0.000001
    )

    for index, is_equal in numpy.ndenumerate(equality):
        if not is_equal:
            print(f"index: {index} value: {is_equal}")
            print(f"image_i = {image_i[index]}, image_i_r = {image_i_r[index]}  ")

    assert numpy.all(equality)


def test_integral_2d():
    w = 31
    h = 71

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu = integral_2d(opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu)
    image_i_r = image_i_gpu.get()

    # pprint(image_o)
    # pprint(image_i)
    # pprint(image_i_r)

    equality = (
        2 * numpy.abs(image_i_r - image_i) / numpy.abs(image_i_r + image_i) < 0.000001
    )

    for index, is_equal in numpy.ndenumerate(equality):
        if not is_equal:
            print(f"index: {index} value: {is_equal}")
            print(f"image_i = {image_i[index]}, image_i_r = {image_i_r[index]}  ")

    assert numpy.all(equality)


def test_integral_3d():
    w = 30
    h = 71
    d = 53

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(d, h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu = integral_3d(opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu)
    image_i_r = image_i_gpu.get()

    # pprint(image_o)
    # pprint(image_i)
    # pprint(image_i_r)

    equality = (
        2 * numpy.abs(image_i_r - image_i) / numpy.abs(image_i_r + image_i) < 0.000001
    )

    for index, is_equal in numpy.ndenumerate(equality):
        if not is_equal:
            print(f"index: {index} value: {is_equal}")
            print(f"image_i = {image_i[index]}, image_i_r = {image_i_r[index]}  ")

    assert numpy.all(equality)


def test_integral_4d():
    w = 30
    h = 71
    d = 53
    e = 12

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(e, d, h, w).astype(numpy.float32)
    image_i = image_o.cumsum(axis=3).cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu = integral_4d(opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu)
    image_i_r = image_i_gpu.get()

    # pprint(image_o)
    # pprint(image_i)
    # pprint(image_i_r)

    equality = (
        2 * numpy.abs(image_i_r - image_i) / numpy.abs(image_i_r + image_i) < 0.000001
    )

    for index, is_equal in numpy.ndenumerate(equality):
        if not is_equal:
            print(f"index: {index} value: {is_equal}")
            print(f"image_i = {image_i[index]}, image_i_r = {image_i_r[index]}  ")

    assert numpy.all(equality)
