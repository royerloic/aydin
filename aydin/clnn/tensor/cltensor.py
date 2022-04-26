import random
from typing import Tuple, Union, Iterable, Any

import numpy
import pyopencl
from numpy.core.multiarray import ndarray
from pyopencl.array import Array, zeros

from aydin.clnn.tensor.opencl.cltensor_kernels import CLTensorKernels
from aydin.clnn.tensor.tensor import Tensor
from aydin.providers.opencl.opencl_provider import OpenCLProvider


class CLTensor(Tensor):
    """
    An OpenCL based implementation of Tensor
    """

    opencl_provider: OpenCLProvider = None

    @classmethod
    def instanciate(cls, shape: Tuple[int, ...], dtype: Any) -> 'NPTensor':
        """
        Constructs a tensor with given shape and dtype.
        """
        dtype = numpy.float32
        tensor = CLTensor(None)
        tensor._clarray = pyopencl.array.zeros(
            CLTensor.opencl_provider.queue, shape, dtype
        )
        return tensor

    def __init__(self, array: Union[ndarray, Iterable, float, int] = None):
        """
        Constructs a tensor.
        """
        super().__init__()

        if array is not None:
            # Ensures that this is a array or right dtype:
            array = numpy.asarray(array, dtype=numpy.float32)
            # Ensure that the order is correct!
            array = numpy.ascontiguousarray(array)

        # Ensures that the opencl provider is allocated at first usage:
        if CLTensor.opencl_provider is None:
            CLTensor.opencl_provider = OpenCLProvider()  # includes='i9-8950HK')

        self._queue = self.opencl_provider.queue

        self._clarray: Array = (
            pyopencl.array.to_device(CLTensor.opencl_provider.queue, array)
            if array is not None
            else None
        )

        self.kernels = CLTensorKernels(CLTensor.opencl_provider)

    def to_class(self, tensor_class):
        """
        Converts tensor to another tensor class
        """
        new_tensor = tensor_class.instanciate(self.shape, numpy.float32)
        new_tensor.nparray = self.nparray
        return new_tensor

    @property
    def size(self) -> int:
        return self._clarray.size

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._clarray.shape

    @property
    def strides(self) -> Tuple[int, ...]:
        return self._clarray.strides

    @property
    def dtype(self):
        return self._clarray.dtype

    @property
    def nparray(self) -> ndarray:
        return self._clarray.get(self._queue)

    @nparray.setter
    def nparray(self, array: Union[ndarray, Iterable, float, int]):
        # Ensures that this is an ndarray of correct dtype
        array = numpy.asarray(array, dtype=numpy.float32)
        # Ensure that the order is correct!
        array = numpy.ascontiguousarray(array)
        self._clarray.set(array, self._queue)

    def load(
        self, array: Union[ndarray, Iterable, float, int, 'CLTensor'], batch_slice
    ) -> 'CLTensor':
        """
        Loads data from array by selecting entries (batch dimension, i.e. axis=0) according to batch_slice
        """
        if isinstance(array, CLTensor):
            self._clarray[...] = array._clarray[batch_slice, ...]
        else:
            self._clarray[...] = array[batch_slice, ...]
        return self

    def sample(
        self, array: 'CLTensor', seed: int = random.randint(0, 2 ** 30)
    ) -> 'CLTensor':
        """
        Samples random entries (batch dimension, i.e. axis=0) from given array and given random seed.
        """
        s = self._clarray
        if not isinstance(array, CLTensor):
            raise NotImplementedError
        a = array._clarray
        self.kernels.sample(s, a, seed)
        return self

    def new(self, shape=None, dtype=None) -> 'CLTensor':
        """
        Instanciates Tensor of same class then self, with given shape and dtype.
        If dtype and/or shape are None, the same shape and dtype as self are used.
        """
        if shape is None:
            shape = self._clarray.shape
        if dtype is None:
            dtype = self._clarray.dtype

        tensor = CLTensor()
        tensor._clarray = zeros(self._queue, shape, dtype)
        return tensor

    def squeeze(self):
        """
        Reshapes tensor so that dimensions of length 1 are removed.
        """
        self._clarray.squeeze()

    def fill(self, value: float) -> 'CLTensor':
        """
        Fills tensor with provided float
        """
        value = numpy.float32(value)
        self._clarray.fill(value)
        return self

    def normal(self, mean: float, std_dev: float) -> 'CLTensor':
        """
        Fills tensor with floats sampled from a normal distribution N(mean, std_dev)
        """
        mean = numpy.float32(mean)
        std_dev = numpy.float32(std_dev)
        nparray = numpy.random.normal(
            loc=mean, scale=std_dev, size=self._clarray.shape
        ).astype(self.dtype)
        self._clarray.set(nparray, self._queue)
        return self

    def dot(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        ta: bool = False,
        tb: bool = False,
        additive=False,
    ) -> 'CLTensor':
        """
        Computes the dot product: self = [self +] a . b
        with optional additivity and transposition of a and b
        """
        s = self._clarray
        a = a._clarray
        b = b._clarray
        self.kernels.dot(s, a, b, ta, tb, additive=additive)
        return self

    def affine(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        c: 'CLTensor',
        ta: bool = False,
        tb: bool = False,
    ) -> 'CLTensor':
        """
        Computes the affine transformation: self = a . b + c
        with optional transposition of a and b
        """
        s = self._clarray
        a = a._clarray
        b = b._clarray
        c = c._clarray
        self.kernels.affine(s, a, b, c, ta, tb, additive=False)
        return self

    def generalised_sum(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        sa: float = 1.0,
        sb: float = 1.0,
        pa: float = 1.0,
        pb: float = 1.0,
        alpha: float = 1.0,
        additive=False,
    ) -> 'CLTensor':
        """
        Computes the barycentric sum with optional power (pa and pb): self = [self + alpha *] ( sa*(a**pa) + sb*(b**pb) )
        with optional additivity.
        """
        sa = numpy.float32(sa)
        sb = numpy.float32(sb)
        pa = numpy.float32(pa)
        pb = numpy.float32(pb)
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray
        self.kernels.generalised_sum(s, a, b, sa, sb, pa, pb, alpha, additive)
        # 's+alpha*(sa*(a**pa)+sb*(b**pb)))'

        return self

    def generalised_product(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        sa: float = 1.0,
        sb: float = 1.0,
        oa: float = 0.0,
        ob: float = 0.0,
        pa: float = 1.0,
        pb: float = 1.0,
        mode: str = 'product',
        alpha: float = 1.0,
        additive=False,
    ) -> 'CLTensor':
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
        s = self._clarray
        a = a._clarray
        b = b._clarray
        self.kernels.generalised_product(
            s, a, b, sa, sb, pa, pb, oa, ob, mode == 'product', alpha, additive
        )
        # f's+alpha*( (sa*(a**pa) + oa) {operation} (sb*(b**pb) + ob) ))',

        return self

    def normalise(
        self, a: 'CLTensor', cut_off: float = 0, alpha: float = 1, additive=False
    ) -> 'CLTensor':
        """
        Normalises (L2) input tensor: self = a/|a|
        """
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        l2norm = self.kernels.l2norm(s)

        if l2norm < cut_off:
            l2norm = 0

        if abs(l2norm) <= 0:
            # we don't normalise zero tensors....
            l2norm = 1.0

        factor = alpha / numpy.float32(l2norm)

        self.kernels.mul_add(s, a, factor, 0.0, additive=False)

        return self

    def clip(self, min_value: float, max_value: float) -> 'CLTensor':
        """
        Clips values between [low, high]
        """
        s = self._clarray
        min_value = numpy.float32(min_value)
        max_value = numpy.float32(max_value)
        self.kernels.clip(s, s, a_min=min_value, a_max=max_value)
        return self

    def copy_from(self, a: 'CLTensor', pad_value: float = 0) -> 'CLTensor':
        """
        Copies between two arrays of possibly different shapes,
        padding and cropping happens when and where needed at the high ends of indices.
        """
        s = self._clarray
        a = a._clarray
        pad_value = numpy.float32(pad_value)
        src_shape = a.shape
        dst_shape = s.shape

        slice_tuple = tuple(slice(0, min(u, v)) for u, v in zip(src_shape, dst_shape))

        if dst_shape[1] > src_shape[1]:
            s.fill(pad_value)

        self.kernels.copy_from(s, a, slice_tuple[1].start, slice_tuple[1].stop)
        # s[slice_tuple] = a[slice_tuple]

        return self

    def noise(self, a: 'CLTensor', noise_level: float = 0) -> 'CLTensor':
        """
        Adds noise: self = a + uniform_noise(noise_level)
        """
        s = self._clarray
        a = a._clarray
        self.kernels.uniform_noise(
            s, a, noise_level=noise_level, seed=random.randint(0, 2 ^ 30)
        )
        return self

    def relu(self, a: 'CLTensor') -> 'CLTensor':
        """
        ReLu: self = relu(a)
        """
        s = self._clarray
        a = a._clarray
        self.kernels.relu(s, a)
        return self

    def abs(self, a: 'CLTensor') -> 'CLTensor':
        """
        Absolute value: self = |a|
        """
        s = self._clarray
        a = a._clarray
        self.kernels.abs(s, a)
        return self

    def signum_select(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        c: 'CLTensor',
        sb: float = 1,
        sc: float = 1,
        alpha: float = 1,
        additive=False,
    ) -> 'CLTensor':
        """
        Computes the sign of a, if a>=0 then selects value: sb * b,
        if a<0 then selects value: sc * c. Note: sa stands for scalar for a.
        An option is given for additivity.
        """
        sb = numpy.float32(sb)
        sc = numpy.float32(sc)
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray
        c = c._clarray
        self.kernels.signum_select(s, a, b, c, sb, sc, alpha, additive)
        #'s+alpha*where(a>=0, sb*b, sc*c)', out=self._clarray)
        return self

    def power_diff(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        p: float = 1.0,
        retain_sign: bool = False,
        alpha: float = 1,
        additive=False,
    ) -> 'CLTensor':
        """
        Computes the absolute power difference: self = |a-b| ** p
        An option is provided to retain the sign of a-b
        with optional additivity.
        """
        one = numpy.float32(1.0)
        p = numpy.float32(p)
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray

        self.kernels.power_diff(s, a, b, p, retain_sign, alpha, additive)
        #'s+where(a>b,one,-one)*alpha*abs(a-b)**p', out=self._clarray)
        #'s+alpha*abs(a-b)**p', out=self._clarray)
        # where(a>b,one,-one)*alpha*abs(a-b)**p', out=self._clarray)
        #'alpha*abs(a-b)**p', out=self._clarray)

        return self

    def squared_diff(
        self,
        a: 'CLTensor',
        b: 'CLTensor',
        retain_sign: bool = False,
        alpha: float = 1,
        additive=False,
    ) -> 'CLTensor':
        """
        Computes the squared difference : self = [self + alpha*] (a-b) ** 2
        with optional additivity, and another option is provided to retain the sign of a-b
        """
        one = numpy.float32(1.0)
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray

        self.kernels.squared_diff(s, a, b, retain_sign, alpha, additive)
        #'s+where(a>b,one,-one)*alpha*(a-b)**2', out=self._clarray)
        #'s+alpha*(a-b)**2', out=self._clarray)
        #'where(a>b,one,-one)*alpha*(a-b)**2', out=self._clarray)
        #'alpha*(a-b)**2', out=self._clarray)
        return self

    def absolute_diff(
        self, a: 'CLTensor', b: 'CLTensor', alpha: float = 1, additive=False
    ) -> 'CLTensor':
        """
        Computes the absolute difference: self = [self + alpha*] abs(a - b)
        with optional additivity.
        """
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray

        self.kernels.absolute_diff(s, a, b, alpha, additive)
        #'s+alpha*abs(a-b)'
        #'alpha*abs(a-b)'
        return self

    def diff_sign(
        self, a: 'CLTensor', b: 'CLTensor', alpha: float = 1, additive=False
    ) -> 'CLTensor':
        """
        Computes the sign of the difference: self = [self + alpha*] signum(a - b)
        with optional additivity.
        """
        one = numpy.float32(1.0)
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray

        self.kernels.diff_sign(s, a, b, alpha, additive)
        #'s+alpha*where(a>b, one, -one)'
        #'alpha*where(a>b, one, -one)'
        return self

    def diff(
        self, a: 'CLTensor', b: 'CLTensor', alpha: float = 1, additive=False
    ) -> 'CLTensor':
        """
        Computes the absolute difference: self = [self + alpha*] (a - b)
        with optional additivity.
        """
        alpha = numpy.float32(alpha)
        s = self._clarray
        a = a._clarray
        b = b._clarray
        alpha = numpy.float32(alpha)

        self.kernels.diff(s, a, b, alpha, additive)
        #'s+alpha*(a-b)'
        #'alpha*(a-b)'
        return self

    def sum(self, a: 'CLTensor', axis: int = None) -> 'CLTensor':
        """
        Computes the sum along an axis: self = sum(a, axis, keepdims)
        """
        s = self._clarray
        a = a._clarray
        self.kernels.sum(s, a, axis=axis)
        return self

    def mean(self, a: 'CLTensor', axis: int = None) -> 'CLTensor':
        """
        Computes the sum along an axis: self = mean(a, axis, keepdims)
        """
        s = self._clarray
        a = a._clarray
        self.kernels.mean(s, a, axis=axis)
        return self

    def __iadd__(self, other) -> 'CLTensor':
        if isinstance(other, CLTensor):
            self._clarray += other._clarray
        else:
            self._clarray += other
        return self

    def __isub__(self, other) -> 'CLTensor':
        if isinstance(other, CLTensor):
            self._clarray -= other._clarray
        else:
            self._clarray -= other
        return self

    def __imul__(self, other) -> 'CLTensor':
        if isinstance(other, CLTensor):
            self._clarray *= other._clarray
        else:
            self._clarray *= other
        return self

    def __itruediv__(self, other) -> 'CLTensor':
        if isinstance(other, CLTensor):
            self._clarray /= other._clarray
        else:
            self._clarray /= other
        return self

    def __ipow__(self, other) -> 'CLTensor':
        if isinstance(other, CLTensor):
            self._clarray **= other._clarray
        else:
            self._clarray **= other
        return self

    def __eq__(self, other) -> bool:
        try:
            if self.shape != other.shape or self.dtype != other.dtype:
                return False
        except AttributeError:
            pass
        if isinstance(other, CLTensor):
            return (self._clarray == other._clarray).all()
        else:
            return (self.nparray == other).all()

    def __ne__(self, other) -> bool:
        try:
            if self.shape != other.shape or self.dtype != other.dtype:
                return True
        except AttributeError:
            pass
        if isinstance(other, CLTensor):
            return (self._clarray != other._clarray).all()
        else:
            return (self.nparray != other).all()

    def sum_all(self) -> float:
        s = self._clarray
        return pyopencl.array.sum(s).get().item(0)

    def mean_all(self) -> float:
        s = self._clarray
        return self.sum_all() / s.size

    def l2_norm(self) -> float:
        s = self._clarray
        return self.kernels.l2norm(s)

    def has_nan_or_inf(self) -> bool:
        s = self._clarray
        return numpy.isnan(s.get()).any() or numpy.isinf(s.get()).any()

    def __str__(self) -> str:
        return self._clarray.__str__()

    def __repr__(self) -> str:
        return self._clarray.__repr__()
