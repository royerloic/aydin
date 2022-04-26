import numpy

from aydin.clnn.modules.special.identity import Identity
from aydin.clnn.modules.special.resize import Resize
from aydin.clnn.tensor.cltensor import CLTensor
from aydin.clnn.tensor.nptensor import NPTensor


def test_identity():
    identity_with(NPTensor)
    identity_with(CLTensor)


def test_crop():
    crop_with(NPTensor)
    crop_with(CLTensor)


def test_pad():
    pad_with(NPTensor)
    pad_with(CLTensor)


def crop_with(tensor_class):
    crop = Resize(shape=(2,))

    x = tensor_class([[0, 1, 2, 3]])
    x_p = crop(x)
    print(x_p)
    assert x_p.shape[1] == 2
    assert x_p.nparray[:, 0:2].sum() == 1

    dx_in = tensor_class([[1, 2]])
    dx_out = crop.backward(dx_in)
    print(dx_out)
    assert (dx_out[0].nparray == [[1, 2, 0, 0]]).all()

    x_np = numpy.random.rand(1024, 16)
    x = tensor_class(x_np)
    print(numpy.where(x.nparray == x_np))
    assert (abs(x.nparray - x_np) < 0.000001).all()


def pad_with(tensor_class):

    pad = Resize(shape=(10,))

    x = tensor_class([[0, 1, 2, 3]])
    x_p = pad(x)
    print(x_p)
    assert x_p.nparray[4:].sum() == 0

    dx_in = tensor_class([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0]])
    dx_out = pad.backward(dx_in)
    print(dx_out)
    assert dx_out[0] == [[0, 1, 2, 3]]


def identity_with(tensor_class):

    identity = Identity()

    x = tensor_class([[0.0, 1, 2, 3]])
    x_p = identity(x)
    print(x_p)
    assert x_p == [[0.0, 1, 2, 3]]

    dx_in = tensor_class([[0.0, 1, 2, 3]])
    dx_out = identity.backward(dx_in)
    print(dx_out)
    assert dx_out[0] == [[0.0, 1, 2, 3]]
