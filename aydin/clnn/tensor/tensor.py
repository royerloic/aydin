import random
from typing import TypeVar, Generic, Tuple, Any, Union, Iterable

from numpy.core.multiarray import ndarray

T = TypeVar('T')


class Tensor(Generic[T]):
    """
    A tensor implements a necessary subset of array and linear algebra operations
    """

    @classmethod
    def instanciate(cls, shape: Tuple[int, ...], dtype: Any) -> T:
        raise NotImplementedError()

    def __init__(self):
        """
        Constructs a tensor.
        """

    def to_class(self, tensor_class):
        """
        Converts tensor to another tensor class
        """
        raise NotImplementedError()

    @property
    def size(self) -> int:
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def strides(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def nparray(self):
        raise NotImplementedError()

    @nparray.setter
    def nparray(self, array: Union[ndarray, Iterable, float, int]):
        raise NotImplementedError()

    def load(self, array: Union[ndarray, Iterable, float, int, T], batch_slice):
        """
        Loads data from array by selecting entries (batch dimension, i.e. axis=0) according to batch_slice
        """
        raise NotImplementedError()

    def sample(
        self,
        array: Union[ndarray, Iterable, float, int, T],
        seed: int = random.randint(0, 2 ** 30),
    ):
        """
        Samples random entries (batch dimension, i.e. axis=0) from given array and given random seed.
        """
        raise NotImplementedError()

    def new(self, shape=None, dtype=None) -> T:
        """
        Instanciates Tensor of same class then self, with given shape and dtype.
        If dtype and/or shape are None, the same shape and dtype as self are used.
        """
        raise NotImplementedError()

    def squeeze(self):
        """
        Reshapes tensor so that dimensions of length 1 are removed.
        """
        raise NotImplementedError()

    def fill(self, value: float) -> T:
        """
        Fills tensor with provided float
        """
        raise NotImplementedError()

    def normal(self, mean: float, std_dev: float) -> T:
        """
        Fills tensor with floats sampled from a normal distribution N(mean, std_dev)
        """
        raise NotImplementedError()

    def dot(self, a: T, b: T, ta: bool = False, tb: bool = False, additive=False) -> T:
        """
        Computes the dot product: self = [self +] a . b
        with optional additivity and transposition of a and b
        """
        raise NotImplementedError()

    def affine(self, a: T, b: T, c: T, ta: bool = False, tb: bool = False) -> T:
        """
        Computes the affine transformation: self = a . b + c
        with optional transposition of a and b
        """
        raise NotImplementedError()

    def generalised_sum(
        self,
        a: T,
        b: T,
        sa: float = 1.0,
        sb: float = 1.0,
        pa: float = 1.0,
        pb: float = 1.0,
        alpha: float = 1.0,
        additive=False,
    ) -> T:
        """
        Computes the generalised sum with optional power (pa and pb): self = sa*(a**pa) + sb*(b**pb)
        """
        raise NotImplementedError()

    def generalised_product(
        self,
        a: T,
        b: T,
        sa: float = 1.0,
        sb: float = 1.0,
        oa: float = 0.0,
        ob: float = 0.0,
        pa: float = 1.0,
        pb: float = 1.0,
        mode: str = 'product',
        alpha: float = 1.0,
        additive=False,
    ) -> T:
        """
        Computes the generalised product of two tensors: self = [self + alpha*] (sa*a**pa+oa) [* or /] (sb*b**pb+ob)
        with optional additivity.
        """
        raise NotImplementedError()

    def normalise(
        self, a: T, cut_off: float = 1e-9, alpha: float = 1, additive=False
    ) -> T:
        """
        Normalises (L2) input tensor: self = a/(|a|^2)
        """
        raise NotImplementedError()

    def clip(self, min_value: float, max_value: float) -> T:
        """
        Clips values between [low, high]
        """
        raise NotImplementedError()

    def copy_from(self, a: T, pad_value: float = 0) -> T:
        """
        Copies between two arrays of possibly different shapes,
        padding and cropping happens when and where needed at the high ends of indices.
        """
        raise NotImplementedError()

    def noise(self, a: T, noise_level: float = 0) -> T:
        """
        Adds noise: self = a + uniform_noise(noise_level)
        """
        raise NotImplementedError()

    def relu(self, a: T) -> T:
        """
        Computes ReLu: self = relu(a)
        """
        raise NotImplementedError()

    def abs(self, a: T) -> T:
        """
        Computes the absolute value: self = abs(a)
        """
        raise NotImplementedError()

    def signum_select(
        self,
        a: T,
        b: T,
        c: T,
        sb: float = 1,
        sc: float = 1,
        alpha: float = 1,
        additive=False,
    ) -> T:
        """
        Computes the sign of a, if a>=0 then selects value: sb * b,
        if a<0 then selects value: sc * c. Note: sa stands for scalar for a
        """

    def power_diff(
        self,
        a: T,
        b: T,
        p: float = 1.0,
        retain_sign: bool = False,
        alpha: float = 1,
        additive=False,
    ) -> T:
        """
        Computes the signed power difference: self = |a-b| ** p
        An option is provided to retain the sign of a-b
        """
        raise NotImplementedError()

    def squared_diff(
        self, a: T, b: T, retain_sign: bool = False, alpha: float = 1, additive=False
    ) -> T:
        """
        Computes the squared difference: self = (a-b)**2
        An option is provided to retain the sign of a-b
        """
        raise NotImplementedError()

    def absolute_diff(self, a: T, b: T, alpha: float = 1, additive=False) -> T:
        """
        Computes the absolute difference: self = abs(a-b)
        """
        raise NotImplementedError()

    def diff_sign(self, a: T, b: T, alpha: float = 1, additive=False) -> T:
        """
        Computes the sign of the difference: self = signum(a - b)
        """
        raise NotImplementedError()

    def diff(self, a: T, b: T, alpha: float = 1, additive=False) -> T:
        """
        Computes the absolute difference: self = abs(a-b)
        """
        raise NotImplementedError()

    def sum(self, a: T, axis: int = None) -> T:
        """
        Computes the sum along an axis: self = sum(a, axis, keepdims)
        """
        raise NotImplementedError()

    def mean(self, a: T, axis: int = None) -> T:
        """
        Computes the sum along an axis: self = mean(a, axis, keepdims)
        """
        raise NotImplementedError()

    def __iadd__(self, other: T) -> T:
        raise NotImplementedError()

    def __isub__(self, other: T) -> T:
        raise NotImplementedError()

    def __imul__(self, other: T) -> T:
        raise NotImplementedError()

    def __itruediv__(self, other: T) -> T:
        raise NotImplementedError()

    def __ipow__(self, other: T) -> T:
        raise NotImplementedError()

    def __eq__(self, other: T) -> bool:
        raise NotImplementedError()

    def __ne__(self, other: T) -> bool:
        raise NotImplementedError()

    def sum_all(self) -> float:
        raise NotImplementedError()

    def mean_all(self) -> float:
        raise NotImplementedError()

    def l2_norm(self) -> float:
        raise NotImplementedError()

    def has_nan_or_inf(self) -> bool:
        raise NotImplementedError()

    def is_zero(self, epsilon=1e-16) -> bool:
        return abs(float(self.mean_all())) < epsilon

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.__str__()
