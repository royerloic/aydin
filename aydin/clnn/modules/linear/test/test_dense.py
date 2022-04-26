import pytest

from aydin.clnn.modules.linear.dense import Dense
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_dense():
    dense_tests_with(NPTensor)
    dense_tests_with(CLTensor)


def dense_tests_with(tensor_class):

    x = tensor_class([[1.1, 2]])

    dense = Dense(nb_inputs=2, nb_outputs=2)
    dense.ensure_parameters_allocated(x)

    dense.weights.nparray = [[1.01, -1.01], [1, -1]]
    dense.biases.nparray = [0.0, 1]

    y = dense(x)

    assert y.nparray[0, 0] == pytest.approx(3.111)
    assert y.nparray[0, 1] == pytest.approx(-2.111)

    x = tensor_class([[1, 0]])
    y = dense.backward(x)[0]
    assert y.nparray[0, 0] == pytest.approx(1.01)
    assert y.nparray[0, 1] == pytest.approx(1.0)
