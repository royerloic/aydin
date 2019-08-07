import numpy
import pyopencl
from pyopencl.array import Array

from aydin.features.fast.integral import (
    integral_1d,
    integral_2d,
    integral_3d,
    integral_4d,
)
from aydin.providers.opencl.opencl_provider import OpenCLProvider


def test_integral_1d():
    n = 32 * 32 * 32 * 32 + 1

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(n).astype(numpy.float32)
    mean = image_o.sum() / image_o.size
    image_i = (image_o - mean).cumsum()

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_i_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu, mean_i = integral_1d(
        opencl_provider, image_o_gpu, image_i_gpu, mean=mean
    )
    image_i_r = image_i_gpu.get()

    average_abs_error = numpy.abs(image_i_r - image_i).sum() / n
    max_abs_error = numpy.max(numpy.abs(image_i_r - image_i))

    print(f"Mean: {mean}")
    print(
        f"Average absolute error: {average_abs_error}, max abs error: {max_abs_error}"
    )

    assert average_abs_error < 0.01
    assert max_abs_error < 0.01


def test_integral_2d():
    w = 32 * 32 + 1
    h = 32 * 32 + 3

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(h, w).astype(numpy.float32)
    mean = image_o.sum() / image_o.size
    image_i = (image_o - mean).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu, _ = integral_2d(
        opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu, mean=mean
    )
    image_i_r = image_i_gpu.get()

    average_abs_error = numpy.abs(image_i_r - image_i).sum() / (w * h)
    max_abs_error = numpy.max(numpy.abs(image_i_r - image_i))

    print(f"Mean: {mean}")
    print(
        f"Average absolute error: {average_abs_error}, max abs error: {max_abs_error}"
    )

    assert average_abs_error < 0.05
    assert max_abs_error < 0.05


def test_integral_3d():
    w = 32 + 1
    h = 32 * 32 + 3
    d = 32 + 5

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(d, h, w).astype(numpy.float32)
    mean = image_o.sum() / image_o.size
    image_i = (image_o - mean).cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu, _ = integral_3d(
        opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu, mean
    )
    image_i_r = image_i_gpu.get()

    average_abs_error = numpy.abs(image_i_r - image_i).sum() / (w * h)
    max_abs_error = numpy.max(numpy.abs(image_i_r - image_i))

    print(f"Mean: {mean}")
    print(
        f"Average absolute error: {average_abs_error}, max abs error: {max_abs_error}"
    )

    assert average_abs_error < 0.5
    assert max_abs_error < 0.5


def test_integral_4d():
    w = 33 + 1
    h = 32 + 3
    d = 32 + 5
    e = 32 + 7

    opencl_provider = OpenCLProvider()

    image_o = numpy.random.rand(e, d, h, w).astype(numpy.float32)
    mean = image_o.sum() / image_o.size
    image_i = (
        (image_o - mean).cumsum(axis=3).cumsum(axis=2).cumsum(axis=1).cumsum(axis=0)
    )

    image_o_gpu = pyopencl.array.to_device(opencl_provider.queue, image_o)
    image_t1_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)
    image_t2_gpu = Array(opencl_provider.queue, image_o_gpu.shape, numpy.float32)

    image_i_gpu, _ = integral_4d(
        opencl_provider, image_o_gpu, image_t1_gpu, image_t2_gpu, mean
    )
    image_i_r = image_i_gpu.get()

    average_abs_error = numpy.abs(image_i_r - image_i).sum() / (w * h)
    max_abs_error = numpy.max(numpy.abs(image_i_r - image_i))

    print(f"Mean: {mean}")
    print(
        f"Average absolute error: {average_abs_error}, max abs error: {max_abs_error}"
    )

    assert average_abs_error < 10
    assert max_abs_error < 10
