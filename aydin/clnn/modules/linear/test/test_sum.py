import pytest

from aydin.clnn.modules.linear.sum import Sum
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_sum():
    sum_tests_with(CLTensor)
    sum_tests_with(NPTensor)


def sum_tests_with(tensor_class):

    # In the case of Sum module, since there can be an arbitrary number of inputs,
    # you have to use 'None' to specify the inputs:
    sum = Sum(Sum(None, None), None)

    u = tensor_class([[1.1, 2]])
    v = tensor_class([[0.9, 2.1]])
    w = tensor_class([[10.1, -2]])

    y = sum((u, v), w)

    print(y)

    assert y.nparray[0, 0] == pytest.approx(12.1)
    assert y.nparray[0, 1] == pytest.approx(2.1)

    dy = tensor_class([[1.0, 0]])
    (du, dv), dw = sum.backward(dy)

    print(du)
    assert du.nparray[0, 0] == pytest.approx(1.0)
    assert du.nparray[0, 1] == pytest.approx(0.0)
    assert dv.nparray[0, 0] == pytest.approx(1.0)
    assert dv.nparray[0, 1] == pytest.approx(0.0)
    assert dw.nparray[0, 0] == pytest.approx(1.0)
    assert dw.nparray[0, 1] == pytest.approx(0.0)
