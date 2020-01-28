from random import randint
from typing import Tuple, Any, Union, Iterable

import numexpr as ne
import numpy
from numpy.core.multiarray import ndarray

from aydin.clnn.tensor.tensor import Tensor


class NPTensor(Tensor['NPTensor']):
    """
    A Numpy Tensor implementation
    """

    @classmethod
    def instanciate(cls, shape: Tuple[int, ...], dtype: Any) -> 'NPTensor':
        """
        Constructs a tensor with given shape and dtype.
        """
        tensor = NPTensor(None)
        tensor._nparray = numpy.zeros(shape, dtype)
        return tensor

    def __init__(self, array: Union[ndarray, Iterable, float, int] = None):
        """
        Constructs a tensor.
        """
        super().__init__()
        self._nparray = numpy.asarray(array) if array is not None else None

    def to_class(self, tensor_class):
        """
        Converts tensor to another tensor class
        """
        new_tensor = tensor_class.instanciate(self.shape, self.dtype)
        new_tensor.nparray = self.nparray.astype(dtype=new_tensor.dtype)
        return new_tensor

    @property
    def size(self) -> int:
        return self._nparray.size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._nparray.shape

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._nparray.strides

    @property
    def dtype(self):
        return self._nparray.dtype

    @property
    def nparray(self) -> ndarray:
        return numpy.copy(self._nparray)

    @nparray.setter
    def nparray(self, array: Union[ndarray, Iterable, float, int]):
        self._nparray[...] = numpy.asarray(array, dtype=self.dtype)

    def load(
        self, array: Union[ndarray, Iterable, float, int, 'NPTensor'], batch_slice
    ) -> 'NPTensor':
        """
        Loads data from array by selecting entries (batch dimension, i.e. axis=0) according to batch_slice
        """
        if isinstance(array, NPTensor):
            self._nparray[...] = array.nparray[batch_slice, ...]
        else:
            self._nparray[...] = array[batch_slice, ...]
        return self

    def sample(
        self,
        array: Union[ndarray, Iterable, float, int, 'NPTensor'],
        seed: int = randint(0, 2 ** 30),
    ) -> 'NPTensor':
        """
        Samples random entries (batch dimension, i.e. axis=0) from given array and given random seed.
        """
        seed_reset = randint(0, 2 ** 30 - 1)
        numpy.random.seed(seed)
        rnd_slice = numpy.random.choice(
            array.shape[0], size=self.shape[0], replace=True
        )
        self.load(array, rnd_slice)
        # we make sure that we don't loose randomness by resetting the seed to an uncorrelated value:
        numpy.random.seed(seed_reset)
        return self

    def new(self, shape=None, dtype=None) -> 'NPTensor':
        """
        Instanciates Tensor of same class then self, with given shape and dtype.
        If dtype and/or shape are None, the same shape and dtype as self are used.
        """
        if shape is None:
            shape = self._nparray.shape
        if dtype is None:
            dtype = self._nparray.dtype

        return NPTensor(numpy.zeros(shape=shape, dtype=dtype))

    def squeeze(self):
        """
        Reshapes tensor so that dimensions of length 1 are removed.
        """
        self._nparray.squeeze()

    def fill(self, value: float) -> 'NPTensor':
        """
        Fills tensor with provided float
        """
        value = numpy.float32(value)
        self._nparray.fill(value)
        return self

    def normal(self, mean: float, std_dev: float) -> 'NPTensor':
        """
        Fills tensor with floats sampled from a normal distribution N(mean, std_dev)
        """
        mean = numpy.float32(mean)
        std_dev = numpy.float32(std_dev)
        self._nparray = numpy.random.normal(
            loc=mean, scale=std_dev, size=self._nparray.shape
        ).astype(numpy.float32)
        return self

    def dot(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        ta: bool = False,
        tb: bool = False,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the dot product: self = [self +] a . b
        with optional additivity and transposition of a and b.
        """
        npa = a._nparray.T if ta else a._nparray
        npb = b._nparray.T if tb else b._nparray

        if additive:
            self._nparray += numpy.dot(npa, npb)
        else:
            numpy.dot(npa, npb, out=self._nparray)

        return self

    def affine(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        c: 'NPTensor',
        ta: bool = False,
        tb: bool = False,
    ) -> 'NPTensor':
        """
        Computes the affine transformation: self = a . b + c
        with optional transposition of a and b
        """
        numpy.dot(a._nparray, b._nparray, out=self._nparray)
        self._nparray += c._nparray
        return self

    def generalised_sum(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        sa: float = 1.0,
        sb: float = 1.0,
        pa: float = 1.0,
        pb: float = 1.0,
        alpha: float = 1.0,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the barycentric sum with optional power (pa and pb): self = [self + alpha *] ( sa*(a**pa) + sb*(b**pb) )
        with optional additivity.
        """
        sa = numpy.float32(sa)
        sb = numpy.float32(sb)
        pa = numpy.float32(pa)
        pb = numpy.float32(pb)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        if additive:
            ne.evaluate('s+alpha*(sa*(a**pa)+sb*(b**pb)))', out=self._nparray)
        else:
            ne.evaluate('alpha*(sa*(a**pa)+sb*(b**pb))', out=self._nparray)
        return self

    def generalised_product(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        sa: float = 1.0,
        sb: float = 1.0,
        oa: float = 0.0,
        ob: float = 0.0,
        pa: float = 1.0,
        pb: float = 1.0,
        mode: str = 'product',
        alpha: float = 1.0,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the generalised product of two tensors: self = [self + alpha*] (sa*a**pa+oa) * (sb*b**pb+ob)
        with optional additivity.
        """
        sa = numpy.float32(sa)
        sb = numpy.float32(sb)
        oa = numpy.float32(oa)
        ob = numpy.float32(ob)
        pa = numpy.float32(pa)
        pb = numpy.float32(pb)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray

        operation: str = '*' if mode == 'product' else (
            '/' if mode == 'division' else 'NOT DEFINED'
        )

        if additive:
            ne.evaluate(
                f's+alpha*( (sa*(a**pa) + oa) {operation} (sb*(b**pb) + ob) ))',
                out=self._nparray,
            )
        else:
            ne.evaluate(
                f'alpha*( (sa*(a**pa) + oa) {operation} (sb*(b**pb) + ob) )',
                out=self._nparray,
            )
        return self

    def normalise(
        self, a: 'NPTensor', cut_off: float = 0, alpha: float = 1, additive=False
    ) -> 'NPTensor':
        """
        Normalises (L2) input tensor: self = a/(|a|^2)
        """
        l2norm = a.l2_norm()
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray

        if l2norm < cut_off:
            l2norm = 0

        if abs(l2norm) <= 0:
            # we don't normalise zero tensors....
            l2norm = 1.0

        l2norm = numpy.float32(l2norm)

        if additive:
            ne.evaluate('s+alpha*(a/l2norm)', out=self._nparray)
        else:
            ne.evaluate('alpha*(a/l2norm)', out=self._nparray)

        return self

    def clip(self, min_value: float, max_value: float) -> 'NPTensor':
        """
        Clips values between [low, high]
        """
        min_value = numpy.float32(min_value)
        max_value = numpy.float32(max_value)
        numpy.clip(self._nparray, a_min=min_value, a_max=max_value, out=self._nparray)
        return self

    def copy_from(self, a: 'NPTensor', pad_value: float = 0) -> 'NPTensor':
        """
        Copies between two arrays of possibly different shapes,
        padding and cropping happens when and where needed at the high ends of indices.
        """
        pad_value = numpy.float32(pad_value)
        src_shape = a._nparray.shape
        dst_shape = self._nparray.shape

        slice_tuple = tuple(slice(0, min(u, v)) for u, v in zip(src_shape, dst_shape))

        self._nparray.fill(pad_value)
        self._nparray[slice_tuple] = a._nparray[slice_tuple]

        return self

    def noise(self, a: 'NPTensor', noise_level: float = 0) -> 'NPTensor':
        """
        Adds noise: self = a + uniform_noise(noise_level)
        """
        s = self._nparray
        a = a._nparray
        noise = (
            numpy.random.uniform(-noise_level, noise_level, 100)
            .reshape(shape=a.shape)
            .astype(a.dtype)
        )
        self._nparray = a + noise

    def relu(self, a: 'NPTensor') -> 'NPTensor':
        """
        ReLu: self = relu(a)
        """
        numpy.maximum(0, a._nparray, out=self._nparray)
        return self

    def abs(self, a: 'NPTensor') -> 'NPTensor':
        """
        Absolute value: self = |a|
        """
        numpy.absolute(a._nparray, out=self._nparray)
        return self

    def signum_select(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        c: 'NPTensor',
        sb: float = 1,
        sc: float = 1,
        alpha: float = 1,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the sign of a, if a>=0 then selects value: sb * b,
        if a<0 then selects value: sc * c. Note: sa stands for scalar for a.
        An option is given for additivity.
        """
        sb = numpy.float32(sb)
        sc = numpy.float32(sc)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        c = c._nparray
        if additive:
            ne.evaluate('s+alpha*where(a>0, sb*b, sc*c)', out=self._nparray)
        else:
            ne.evaluate('alpha*where(a>0, sb*b, sc*c)', out=self._nparray)
        return self

    def power_diff(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        p: float = 1.0,
        retain_sign: bool = False,
        alpha: float = 1,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the absolute power difference: self = |a-b| ** p
        An option is provided to retain the sign of a-b
        with optional additivity.
        """
        one = numpy.float32(1.0)
        p = numpy.float32(p)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray

        if additive:
            if retain_sign:
                ne.evaluate(
                    's+where(a>b,one,-one)*alpha*abs(a-b)**p', out=self._nparray
                )
            else:
                ne.evaluate('s+alpha*abs(a-b)**p', out=self._nparray)
        else:
            if retain_sign:
                ne.evaluate('where(a>b,one,-one)*alpha*abs(a-b)**p', out=self._nparray)
            else:
                ne.evaluate('alpha*abs(a-b)**p', out=self._nparray)
        return self

    def squared_diff(
        self,
        a: 'NPTensor',
        b: 'NPTensor',
        retain_sign: bool = False,
        alpha: float = 1,
        additive=False,
    ) -> 'NPTensor':
        """
        Computes the squared difference : self = [self + alpha*] (a-b) ** 2
        with optional additivity, and another option is provided to retain the sign of a-b
        """
        one = numpy.float32(1.0)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        if additive:
            if retain_sign:
                ne.evaluate('s+where(a>b,one,-one)*alpha*(a-b)**2', out=self._nparray)
            else:
                ne.evaluate('s+alpha*(a-b)**2', out=self._nparray)
        else:
            if retain_sign:
                ne.evaluate('where(a>b,one,-one)*alpha*(a-b)**2', out=self._nparray)
            else:
                ne.evaluate('alpha*(a-b)**2', out=self._nparray)
        return self

    def absolute_diff(
        self, a: 'NPTensor', b: 'NPTensor', alpha: float = 1, additive=False
    ) -> 'NPTensor':
        """
        Computes the absolute difference: self = [self + alpha*] abs(a - b)
        with optional additivity.
        """
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        if additive:
            ne.evaluate('s+alpha*abs(a-b)', out=self._nparray)
        else:
            ne.evaluate('alpha*abs(a-b)', out=self._nparray)
        return self

    def diff_sign(
        self, a: 'NPTensor', b: 'NPTensor', alpha: float = 1, additive=False
    ) -> 'NPTensor':
        """
        Computes the sign of the difference: self = [self + alpha*] signum(a - b)
        with optional additivity.
        """
        one = numpy.float32(1.0)
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        if additive:
            ne.evaluate('s+alpha*where(a>b, one, -one)', out=self._nparray)
        else:
            ne.evaluate('alpha*where(a>b, one, -one)', out=self._nparray)
        return self

    def diff(
        self, a: 'NPTensor', b: 'NPTensor', alpha: float = 1, additive=False
    ) -> 'NPTensor':
        """
        Computes the absolute difference: self = [self + alpha*] (a - b)
        with optional additivity.
        """
        alpha = numpy.float32(alpha)
        s = self._nparray
        a = a._nparray
        b = b._nparray
        alpha = numpy.float32(alpha)
        if additive:
            ne.evaluate('s+alpha*(a-b)', out=self._nparray)
        else:
            ne.evaluate('alpha*(a-b)', out=self._nparray)
        return self

    def sum(self, a: 'NPTensor', axis: int = None) -> 'NPTensor':
        """
        Computes the sum along an axis: self = sum(a, axis, keepdims)
        """
        if axis is None:
            axis = numpy._NoValue
        keepdims = len(self.shape) == len(a.shape)
        numpy.sum(a._nparray, axis=axis, out=self._nparray, keepdims=keepdims)
        return self

    def mean(self, a: 'NPTensor', axis: int = None) -> 'NPTensor':
        """
        Computes the sum along an axis: self = mean(a, axis, keepdims)
        """
        if axis is None:
            axis = numpy._NoValue
        keepdims = len(self.shape) == len(a.shape)
        numpy.mean(a._nparray, axis=axis, out=self._nparray, keepdims=keepdims)
        return self

    def __iadd__(self, other) -> 'NPTensor':
        if isinstance(other, NPTensor):
            self._nparray += other._nparray
        else:
            self._nparray += other
        return self

    def __isub__(self, other) -> 'NPTensor':
        if isinstance(other, NPTensor):
            self._nparray -= other._nparray
        else:
            self._nparray -= other
        return self

    def __imul__(self, other) -> 'NPTensor':
        if isinstance(other, NPTensor):
            self._nparray *= other._nparray
        else:
            self._nparray *= other
        return self

    def __itruediv__(self, other) -> 'NPTensor':
        if isinstance(other, NPTensor):
            self._nparray /= other._nparray
        else:
            self._nparray /= other
        return self

    def __ipow__(self, other) -> 'NPTensor':
        if isinstance(other, NPTensor):
            self._nparray **= other._nparray
        else:
            self._nparray **= other
        return self

    def __eq__(self, other) -> bool:
        try:
            if self.shape != other.shape or self.dtype != other.dtype:
                return False
        except AttributeError:
            pass
        if isinstance(other, NPTensor):
            return (self._nparray == other._nparray).all()
        else:
            return (self._nparray == other).all()

    def __ne__(self, other) -> bool:
        try:
            if self.shape != other.shape or self.dtype != other.dtype:
                return True
        except AttributeError:
            pass
        if isinstance(other, NPTensor):
            return (self._nparray != other._nparray).all()
        else:
            return (self._nparray != other).all()

    def sum_all(self) -> float:
        s = self._nparray
        return s.sum()

    def mean_all(self) -> float:
        s = self._nparray
        return s.mean()

    def l2_norm(self) -> float:
        s = self._nparray
        return numpy.linalg.norm(s, ord=2)

    def has_nan_or_inf(self) -> bool:
        s = self._nparray
        return numpy.isnan(s).any() or numpy.isinf(s).any()

    def __str__(self) -> str:
        return self._nparray.__str__()

    def __repr__(self) -> str:
        return self._nparray.__repr__()
