import pytest

from aydin.clnn.modules.nonlin.abs import Abs
from aydin.clnn.modules.nonlin.relu import ReLU
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_abs():
    abs_with(NPTensor)
    abs_with(CLTensor)


def test_relu():
    relu_with(NPTensor)
    relu_with(CLTensor)


def abs_with(tensor_class):

    abs = Abs()

    x = tensor_class([[0.0, -1, 2, -3]])
    x_p = abs(x)
    print(x_p)
    assert x_p.shape[1] == 4
    assert x_p.nparray[0, 0] == 0.0
    assert x_p.nparray[0, 1] == 1.0
    assert x_p.nparray[0, 2] == 2.0
    assert x_p.nparray[0, 3] == 3.0

    dx_in = tensor_class([[1.0, 1, -1, -2]])
    dx_out = abs.backward(dx_in)
    print(dx_out)
    assert (dx_out[0].nparray == [[-1.0, -1, -1, 2]]).all()


def relu_with(tensor_class):

    relu = ReLU()

    x = tensor_class([[0.0, -1, 2, -3]])
    x_p = relu(x)
    print(x_p)
    assert x_p.shape[1] == 4
    assert x_p.nparray[0, 0] == 0.0
    assert x_p.nparray[0, 1] == 0
    assert x_p.nparray[0, 2] == 2.0
    assert x_p.nparray[0, 3] == 0

    dx_in = tensor_class([[1.0, 1, -1, -2]])
    dx_out = relu.backward(dx_in)
    print(dx_out)
    assert dx_out[0].nparray[0, 0] == pytest.approx(0.0, abs=0.001)
    assert (dx_out[0].nparray[0, 1] < 0) and (dx_out[0].nparray[0, 1] >= -0.001)
    assert dx_out[0].nparray[0, 2] == -1.0
    assert (dx_out[0].nparray[0, 3] > 0) and (dx_out[0].nparray[0, 3] <= 0.001)
