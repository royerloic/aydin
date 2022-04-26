from pprint import pprint

import numpy
import pytest

from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_basics():
    basics_with(CLTensor)
    basics_with(NPTensor)


def basics_with(tensor_class, advanced=True):
    """
    Testing the basics of tensors using the given class for instanciating the tensors
    """

    CLTensor.opencl_provider = None
    # testing new:
    x = tensor_class([[0.0, 1], [1, 0]])
    assert x.nparray[0, 0] == 0.0
    assert x.nparray[0, 1] == 1.0
    assert x.nparray[1, 0] == 1
    assert x.nparray[1, 1] == 0.0

    assert x.new().shape == (2, 2)
    assert x.new().dtype == numpy.float64 or x.new().dtype == numpy.float32

    # testing nparray:
    x = tensor_class([[0.0, 0.1], [0.1, 0]])
    y = x.new()
    y.nparray = numpy.asarray([[1, 1], [1, 1.1]])
    assert y.nparray[0, 0] == 1
    assert y.nparray[1, 1] == pytest.approx(1.1, rel=0.01)

    # testing nparray more:
    x_np = numpy.random.rand(1024, 16)
    x = tensor_class(x_np)
    y = x.new()
    y.nparray = x_np
    assert (abs(x.nparray - x_np) < 0.000001).all()
    assert (abs(y.nparray - x_np) < 0.000001).all()

    # testing to_tensor_class:
    x_np = numpy.random.rand(1024, 16)
    x = tensor_class(x_np)
    y1 = x.to_class(CLTensor)
    y2 = x.to_class(NPTensor)

    assert (x.nparray - y1.nparray) == pytest.approx(0.0, abs=0.001)
    assert (x.nparray - y2.nparray) == pytest.approx(0.0, abs=0.001)
    assert (y1.nparray - y2.nparray) == pytest.approx(0.0, abs=0.001)

    # testing sample seeds:
    x_np = numpy.random.rand(1024, 16)
    x = tensor_class(x_np)
    y = x.new(shape=(10, 16))
    z = x.new(shape=(10, 16))
    u = x.new(shape=(10, 16))
    for i in range(0, 10):
        y.sample(x, 13 + i * 17)
        z.sample(x, 13 + i * 17)
        assert y == z
        u.sample(x, i)
        assert y != u
        assert z != u

    # testing sample coverage:
    length = 100
    x_np = numpy.arange(length)[:, numpy.newaxis]
    x = tensor_class(x_np)
    y = x.new(shape=(17, 1))
    bag = set()
    for i in range(0, 100):
        y.sample(x, i)
        for j in range(0, 17):
            bag.add(y.nparray[j, 0])
    # print(bag)
    assert len(bag) > length // 2

    # testing sample pattern:
    length = 64
    x_np = numpy.arange(length)[:, numpy.newaxis]
    x = tensor_class(x_np)
    y = x.new(shape=(length // 2, 1))
    y.sample(x, 17)
    pprint(y)
    y.sample(x, 18)
    pprint(y)

    # testing fill:
    x = tensor_class([[0.0, 1], [1, 0]])
    x.fill(1.3)
    assert x.nparray.sum() == pytest.approx(4 * 1.3)

    # testing normal:
    x = tensor_class([[1.3, 1.3], [1.3, 1.3]])
    assert x.nparray.sum() == pytest.approx(4 * 1.3, rel=0.01)
    x.normal(mean=1.4, std_dev=2.9)
    assert x.nparray.sum() != pytest.approx(4 * 1.3, rel=0.01)

    if advanced:

        # testing affine:
        a = tensor_class([[0.0, 1, 2], [1, 0, 1]])
        b = tensor_class([[0.0, 1.5, 0, 0], [2.0, 3.5, 0, 0], [4.0, 5.5, 0, 0]])
        c = tensor_class([0.1, 0.2, 0.3, 0.4])
        w = c.new(shape=(2, 4))
        w.affine(a, b, c)
        assert w.nparray[0, 0] == pytest.approx(10.1)
        assert w.nparray[0, 1] == pytest.approx(14.7)
        assert w.nparray[0, 2] == pytest.approx(0.3)
        assert w.nparray[0, 3] == pytest.approx(0.4)
        assert w.nparray[1, 0] == pytest.approx(4.1)
        assert w.nparray[1, 1] == pytest.approx(7.2)
        assert w.nparray[1, 2] == pytest.approx(0.3)
        assert w.nparray[1, 3] == pytest.approx(0.4)

        # testing affine:
        a = tensor_class([[0.0, 1, 2], [1, 0, 1]])
        b = tensor_class([[0.0, 1.5, 0, 0], [2.0, 3.5, 0, 0], [4.0, 5.5, 0, 0]])
        c = tensor_class([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
        w = c.new(shape=(2, 4))
        w.affine(a, b, c)
        assert w.nparray[0, 0] == pytest.approx(10.1)
        assert w.nparray[0, 1] == pytest.approx(14.6)
        assert w.nparray[0, 2] == pytest.approx(0.1)
        assert w.nparray[0, 3] == pytest.approx(0.1)
        assert w.nparray[1, 0] == pytest.approx(4.1)
        assert w.nparray[1, 1] == pytest.approx(7.1)
        assert w.nparray[1, 2] == pytest.approx(0.1)
        assert w.nparray[1, 3] == pytest.approx(0.1)

        # testing affine again:
        x = tensor_class([[0.0, 1], [1, 0]])
        m = tensor_class([[0, 1.0], [1, 0]])
        y = m.new()
        y.affine(x, m, x)
        assert y.nparray[0, 0] == y.nparray[0, 1]
        assert y.nparray[0, 1] == y.nparray[0, 0]
        assert y.nparray[1, 0] == y.nparray[1, 1]
        assert y.nparray[1, 1] == y.nparray[1, 0]

        # testing affine again:
        a = tensor_class([[0.0, 1], [1, 0]])
        x = tensor_class([[2.0, 0.5]])
        b = tensor_class([[-1.3, 1.8]])
        w = m.new(shape=(1, 2))
        w.affine(x, a, b)
        assert w.nparray[0, 0] == pytest.approx(-0.8)
        assert w.nparray[0, 1] == pytest.approx(3.8)

        # testing dot:
        x = tensor_class([[0.0, 1], [2, 0]])
        m = tensor_class([[0, 1.0], [1, 0]])
        y = m.new()
        y.dot(x, m)
        assert x.nparray[0, 0] == y.nparray[0, 1]
        assert x.nparray[0, 1] == y.nparray[0, 0]
        assert x.nparray[1, 0] == y.nparray[1, 1]
        assert x.nparray[1, 1] == pytest.approx(2 * y.nparray[1, 0])

        # testing dot again:
        u = tensor_class([[0.0, 1], [1, 0]])
        v = tensor_class([[2.0, 0.5], [-0.3, 0.7]])
        w = u.new(shape=(2, 2))
        w.dot(u, v)
        assert w.nparray[0, 0] == pytest.approx(-0.3)
        assert w.nparray[0, 1] == pytest.approx(0.7)
        assert w.nparray[1, 0] == pytest.approx(2.0)
        assert w.nparray[1, 1] == pytest.approx(0.5)

        # testing dot with vector:
        m = tensor_class([[0.0, 1], [1, 0]])
        x = tensor_class([[2.0, 0.5]])
        w = m.new(shape=(1, 2))
        w.dot(x, m)
        assert w.nparray[0, 0] == pytest.approx(0.5, rel=0.01)
        assert w.nparray[0, 1] == pytest.approx(2.0, rel=0.01)

        # testing dot with transposed matrix:
        m = tensor_class([[0.0, 1], [2, 0]])
        x = tensor_class([[2.0, 0.5]])
        w = m.new(shape=(1, 2))
        w.dot(x, m, tb=True)
        assert w.nparray[0, 0] == pytest.approx(0.5, rel=0.01)
        assert w.nparray[0, 1] == pytest.approx(4.0, rel=0.01)

        # testing dot with transposed matrices:
        a = tensor_class([[0.0, 2], [1, 0], [-1, 1]])
        b = tensor_class([[1.0, 2, 3], [4, 5, 6]])
        w = m.new(shape=(2, 2))
        w.dot(a, b, ta=True, tb=True)
        assert w.nparray[0, 0] == pytest.approx(-1.0, rel=0.01)
        assert w.nparray[0, 1] == pytest.approx(-1.0, rel=0.01)
        assert w.nparray[1, 0] == pytest.approx(5, rel=0.01)
        assert w.nparray[1, 1] == pytest.approx(14, rel=0.01)

    # testing generalised_sum:
    a = tensor_class([[0.0, 1], [1, 0]])
    b = tensor_class([[-1.3, 1.8], [0, 0]])
    w = a.new(shape=(2, 2))
    w.generalised_sum(a, b)
    assert w.nparray[0, 0] == pytest.approx(-1.3)
    assert w.nparray[0, 1] == pytest.approx(2.8)
    assert w.nparray[1, 0] == pytest.approx(1.0)
    assert w.nparray[1, 1] == pytest.approx(0.0)

    # testing generalised_product:
    a = tensor_class([[0.0, 1], [1, 0]])
    b = tensor_class([[-1.3, 1.8], [0, 0]])
    w = a.new(shape=(2, 2))
    w.generalised_product(a, b)
    assert w.nparray[0, 0] == pytest.approx(-0.0)
    assert w.nparray[0, 1] == pytest.approx(1.8)
    assert w.nparray[1, 0] == pytest.approx(0.0)
    assert w.nparray[1, 1] == pytest.approx(0.0)

    # testing normalise:
    x = tensor_class([[0.0, 1], [1, 0]])
    x.normal(mean=1.4, std_dev=2.9)
    assert x.l2_norm() != pytest.approx(1.0)
    x.normalise(x)
    # TODO: Why is this not working?
    # assert x.l2_norm() == pytest.approx(1.0, abs=0.1)

    # testing clip:
    y.normal(mean=0.5, std_dev=20.9)
    y.clip(0, 1)
    assert (y.nparray <= 1).all()
    assert (0 <= y.nparray).all()

    # testing copy_from:
    u = tensor_class([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = u.new(shape=(u.shape[0] + 1, u.shape[1] - 1))
    y.fill(0.0)
    y.copy_from(u)
    assert y.nparray[0, 0] == pytest.approx(1.0)
    assert y.nparray[0, 1] == pytest.approx(2.0)
    assert y.nparray[1, 0] == pytest.approx(4.0)
    assert y.nparray[1, 1] == pytest.approx(5.0)
    assert y.nparray[2, 0] == pytest.approx(0.0)
    assert y.nparray[2, 1] == pytest.approx(0.0)

    # testing relu:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    y = x.new()
    y.relu(x)
    assert y.nparray[0, 0] == pytest.approx(10.0)
    assert y.nparray[0, 1] == pytest.approx(0)
    assert y.nparray[1, 0] == pytest.approx(0)
    assert y.nparray[1, 1] == pytest.approx(0.1)

    # testing abs:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    y = x.new()
    y.abs(x)
    assert y.nparray[0, 0] == pytest.approx(10.0)
    assert y.nparray[0, 1] == pytest.approx(1.0)
    assert y.nparray[1, 0] == pytest.approx(0.1)
    assert y.nparray[1, 1] == pytest.approx(0.1)

    # testing signum_select:
    a = tensor_class([[0.0, -1], [0.1, -10]])
    b = tensor_class([[1.2, 2.3], [3.4, 4.5]])
    c = tensor_class([[1.1, 2.2], [3.3, 4.4]])
    w = a.new(shape=(2, 2))
    w.signum_select(a, b, c)
    assert w.nparray[0, 0] == pytest.approx(1.1)
    assert w.nparray[0, 1] == pytest.approx(2.2)
    assert w.nparray[1, 0] == pytest.approx(3.4)
    assert w.nparray[1, 1] == pytest.approx(4.4)

    # testing power_diff:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    m = tensor_class([[0, 1.0], [1, 0]])
    y.power_diff(x, m, p=0.5)
    assert y.nparray[0, 0] == pytest.approx(3.16, abs=0.01)
    assert y.nparray[0, 1] == pytest.approx(1.41, abs=0.01)
    assert y.nparray[1, 0] == pytest.approx(1.04, abs=0.01)
    assert y.nparray[1, 1] == pytest.approx(0.31, abs=0.01)

    # testing squared_diff:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    m = tensor_class([[0, 1.0], [1, 0]])
    y.squared_diff(x, m)
    assert y.nparray[0, 0] == pytest.approx(100.0)
    assert y.nparray[0, 1] == pytest.approx(4.0)

    # testing absolute_diff:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    m = tensor_class([[0, 1.0], [1, 0]])
    y.absolute_diff(x, m)
    assert y.nparray[0, 0] == pytest.approx(10.0)
    assert y.nparray[0, 1] == pytest.approx(2.0)

    # testing diff:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    m = tensor_class([[0, 1.0], [1, 0]])
    y.diff(x, m)
    assert y.nparray[0, 0] == pytest.approx(10.0)
    assert y.nparray[0, 1] == pytest.approx(-2.0)

    # testing sum:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    y = m.new(shape=(2, 1))
    y.sum(x, axis=1)
    assert y.nparray[0, 0] == pytest.approx(9.0)
    assert y.nparray[1, 0] == pytest.approx(0.0)

    # testing mean:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    y = m.new(shape=(2, 1))
    y.mean(x, axis=1)
    assert y.nparray[0, 0] == pytest.approx(4.5)
    assert y.nparray[1, 0] == pytest.approx(0.0)

    # testing +=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    x += 1
    assert x.nparray[0, 0] == pytest.approx(11.0)
    assert x.nparray[1, 0] == pytest.approx(0.9)

    # testing -=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    x -= 1
    assert x.nparray[0, 0] == pytest.approx(9.0)
    assert x.nparray[1, 1] == pytest.approx(-0.9)

    # testing /=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    x /= 2.0
    assert x.nparray[0, 0] == pytest.approx(5.0)
    assert x.nparray[1, 1] == pytest.approx(0.05)

    # testing *=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    x *= 2
    assert x.nparray[0, 0] == pytest.approx(20.0)
    assert x.nparray[1, 1] == pytest.approx(0.2)

    # testing **=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    x **= 2.0
    assert x.nparray[0, 0] == pytest.approx(100.0)
    assert x.nparray[0, 1] == pytest.approx(1.0)
    assert x.nparray[1, 0] == pytest.approx(0.01)
    assert x.nparray[1, 1] == pytest.approx(0.01)

    # testing == and !=:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    y = tensor_class([[5.0, -0.5], [-0.05, 0.05]])
    assert x != y
    y *= 2
    assert x == y

    # testing sum_all:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    assert x.sum_all() == pytest.approx(9.0)

    # testing mean_all:
    x = tensor_class([[10.0, -1], [-0.1, 0.1]])
    assert x.mean_all() == pytest.approx(2.25)

    # testing has_nan_or_inf:
    x = tensor_class([[10.0, -1], [-0.1, 1.0]])
    assert not x.has_nan_or_inf()
    x = tensor_class([[10.0, -1], [-0.1, numpy.math.nan]])
    assert x.has_nan_or_inf()
    x = tensor_class([[10.0, -1], [-0.1, numpy.math.inf]])
    assert x.has_nan_or_inf()
