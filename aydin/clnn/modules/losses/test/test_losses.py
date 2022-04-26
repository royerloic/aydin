import pytest

from aydin.clnn.modules.losses.l1 import L1Loss
from aydin.clnn.modules.losses.l2 import L2Loss
from aydin.clnn.modules.losses.lh import LHalfLoss
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_lh():
    lh_with(NPTensor)
    lh_with(CLTensor)


def test_l1():
    l1_with(NPTensor)
    l1_with(CLTensor)


def test_l2():
    l2_with(NPTensor)
    l2_with(CLTensor)


def lh_with(tensor_class):

    lh = LHalfLoss()

    x = tensor_class([[0.0, -1.00001, 2, -3]])
    xt = tensor_class([[-0.1, -1, 2.1, 1]])

    y = lh(x, xt)
    print(y)

    assert y.nparray[0, 0] == pytest.approx(0.1 ** 0.5)
    assert y.nparray[0, 1] == pytest.approx(0.003, abs=1e-2)
    assert y.nparray[0, 2] == pytest.approx(0.1 ** 0.5)
    assert y.nparray[0, 3] == pytest.approx(4 ** 0.5)

    dy = tensor_class([[0.0, -1, 1, -1]])
    dx = lh.backward(dy)

    print(dx)

    assert dx[0].nparray[0, 0] == pytest.approx(1.0)
    assert dx[0].nparray[0, 1] == pytest.approx(-1.0)
    assert dx[0].nparray[0, 2] == pytest.approx(-1.0)
    assert dx[0].nparray[0, 3] == pytest.approx(-0.25)


def l1_with(tensor_class):

    l1 = L1Loss()

    x = tensor_class([[0.0, -1.00002, 2, -3]])
    xt = tensor_class([[-0.1, -1, 2.2, 1]])

    y = l1(x, xt)
    print(y)

    assert y.nparray[0, 0] == pytest.approx(0.1)
    assert y.nparray[0, 1] == pytest.approx(2.0e-5, rel=0.01)
    assert y.nparray[0, 2] == pytest.approx(0.2)
    assert y.nparray[0, 3] == pytest.approx(4)

    dy = tensor_class([[0.0, -1, 1, -1]])
    dx = l1.backward(dy)

    print(dx)

    assert dx[0].nparray[0, 0] == pytest.approx(1.0)
    assert dx[0].nparray[0, 1] == pytest.approx(-1.0)
    assert dx[0].nparray[0, 2] == pytest.approx(-1.0)
    assert dx[0].nparray[0, 3] == pytest.approx(-1.0)


def l2_with(tensor_class):

    l2 = L2Loss()

    x = tensor_class([[0.0, -1.00002, 2, -3]])
    xt = tensor_class([[-0.1, -1, 2.2, 1]])

    y = l2(x, xt)
    print(y)

    assert y.nparray[0, 0] == pytest.approx(0.01)
    assert y.nparray[0, 1] == pytest.approx(4.0e-10, rel=0.01)
    assert y.nparray[0, 2] == pytest.approx(0.04)
    assert y.nparray[0, 3] == pytest.approx(16)

    dy = tensor_class([[0.0, -1, 1, -1]])
    dx = l2.backward(dy)

    print(dx)
    assert len(dx) == 1

    assert dx[0].nparray[0, 0] == pytest.approx(0.2)
    assert dx[0].nparray[0, 1] == pytest.approx(-4.0e-5, rel=0.01)
    assert dx[0].nparray[0, 2] == pytest.approx(-4e-1)
    assert dx[0].nparray[0, 3] == pytest.approx(-8)
