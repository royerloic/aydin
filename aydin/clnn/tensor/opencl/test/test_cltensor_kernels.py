import pytest

from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.opencl.cltensor_kernels import CLTensorKernels
from aydin.providers.opencl.opencl_provider import OpenCLProvider


def test_sum_kernel():

    provider = OpenCLProvider()
    CLTensor.opencl_provider = provider
    kernels = CLTensorKernels(provider)

    # sum of 1d array without axis defined:
    x = CLTensor([0.0, 1, 1, 0])
    y = x.new(shape=(1, 1))
    kernels.sum(y._clarray, x._clarray)
    assert y.nparray[0, 0] == 2.0

    # sum of 2d array without axis defined and keeping dimensions:
    x = CLTensor([[0.0, 1], [1, 0]])
    y = x.new(shape=(1, 1))
    kernels.sum(y._clarray, x._clarray)
    assert y.nparray[0, 0] == 2.0

    # sum of 2d array with axis defined and keeping dimensions:
    x = CLTensor([[0.0, 1], [1, 0], [2, 3]])
    y = x.new(shape=(1, 2))
    kernels.sum(y._clarray, x._clarray, axis=0)
    assert y.nparray[0, 0] == 3.0
    assert y.nparray[0, 1] == 4.0

    x = CLTensor([[0.0, 1], [1, 0], [2, 3]])
    y = x.new(shape=(3, 1))
    kernels.sum(y._clarray, x._clarray, axis=1)
    assert y.nparray[0, 0] == 1.0
    assert y.nparray[1, 0] == 1.0
    assert y.nparray[2, 0] == 5.0

    provider.queue.finish()
    del provider


def test_mean_kernel():

    provider = OpenCLProvider()
    CLTensor.opencl_provider = provider
    kernels = CLTensorKernels(provider)

    # sum of 1d array without axis defined:
    x = CLTensor([0.0, 1, 1, 0])
    y = x.new(shape=(1, 1))
    kernels.mean(y._clarray, x._clarray)
    assert y.nparray[0, 0] == 0.5

    # sum of 2d array without axis defined and keeping dimensions:
    x = CLTensor([[0.0, 1], [1, 0]])
    y = x.new(shape=(1, 1))
    kernels.mean(y._clarray, x._clarray)
    assert y.nparray[0, 0] == 0.5

    # sum of 2d array with axis defined and keeping dimensions:
    x = CLTensor([[0.0, 1], [1, 0], [2, 3]])
    y = x.new(shape=(1, 2))
    kernels.mean(y._clarray, x._clarray, axis=0)
    assert y.nparray[0, 0] == 1.0
    assert y.nparray[0, 1] == pytest.approx(1.3333334)

    x = CLTensor([[0.0, 1], [1, 0], [2, 3]])
    y = x.new(shape=(3, 1))
    kernels.mean(y._clarray, x._clarray, axis=1)
    assert y.nparray[0, 0] == 0.5
    assert y.nparray[1, 0] == 0.5
    assert y.nparray[2, 0] == 2.5

    provider.queue.finish()
    del provider


def test_copy_from_kernel():

    provider = OpenCLProvider()
    CLTensor.opencl_provider = provider
    kernels = CLTensorKernels(provider)

    x = CLTensor([[-4.0, -3, -2, -1], [1.0, 2, 3, 4], [5.0, 5, 6, 8]])
    y = x.new(shape=(3, 2))

    kernels.copy_from(y._clarray, x._clarray, 1, 3)

    assert y.nparray[0, 0] == -3.0
    assert y.nparray[0, 1] == -2.0
    assert y.nparray[1, 0] == 2.0
    assert y.nparray[1, 1] == 3.0
    assert y.nparray[2, 0] == 5.0
    assert y.nparray[2, 1] == 6.0

    provider.queue.finish()
    del provider
