import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle
import numexpr
import numpy

from aydin.util.json import encode_indent
from aydin.util.log.log import lprint


class NormaliserBase(ABC):
    """
        Normaliser base class

    """

    epsilon: float
    leave_as_float: bool
    clip: bool
    original_dtype: numpy.dtype

    def __init__(self, clip=True, leave_as_float=False, epsilon=1e-21):
        """
        Constructs a normaliser
        """
        self.epsilon = epsilon
        self.clip = clip
        self.leave_as_float = leave_as_float

        self.rmin = None
        self.rmax = None

    def save(self, path: str, name='default'):
        """
        Saves an 'all-batteries-included' normaliser at a given path (folder).
        :param path: path to save to
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving normaliser to: {path}")
        with open(join(path, f"normaliser_{name}.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str, name='default'):
        """
        Returns an 'all-batteries-included' normaliser from a given path (folder).
        :param model_path: path to load from.
        """

        lprint(f"Loading normaliser from: {path}")
        with open(join(path, f"normaliser_{name}.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        return thawed

    @abstractmethod
    def calibrate(self, array):
        """
        Calibrates this normaliser given an array.

        :param array: array to use for calibration
        :type array: ndarray
        """
        raise NotImplementedError()

    def normalise(self, array):
        """
        Normalises the given array in-place (if possible).

        :param array: array to normaliser
        :type array: ndarray
        """
        if array.dtype != numpy.float32:
            array = array.astype(numpy.float32)

        if self.rmin is not None and self.rmax is not None:
            min_value = numpy.float32(self.rmin)
            max_value = numpy.float32(self.rmax)
            epsilon = numpy.float32(self.epsilon)

            try:
                # We perform operation in-place with numexpr if possible:
                numexpr.evaluate(
                    "(array - min_value) / ( max_value - min_value + epsilon )",
                    out=array,
                )
                if self.clip:
                    numexpr.evaluate(
                        "where(array<0,0,where(array>1,1,array))", out=array
                    )

            except ValueError:
                array -= min_value
                array /= max_value - min_value + epsilon
                if self.clip:
                    array = numpy.clip(array, 0, 1)  # , out=array

        return array

    def denormalise(self, array: numpy.ndarray):
        """
        Denormalises the given array in-place (if possible).
        :param array: array to denormalise
        :type array: ndarray
        """
        if self.rmin is not None and self.rmax is not None:

            min_value = numpy.float32(self.rmin)
            max_value = numpy.float32(self.rmax)
            epsilon = numpy.float32(self.epsilon)

            try:
                # We perform operation in-place with numexpr if possible:
                if self.clip:
                    numexpr.evaluate(
                        "where(array<0,0,where(array>1,1,array))", out=array
                    )
                numexpr.evaluate(
                    "array * (max_value - min_value + epsilon) + min_value ", out=array
                )

            except ValueError:
                if self.clip:
                    array = numpy.clip(array, 0, 1)  # , out=array
                array *= max_value - min_value + epsilon
                array += min_value

        if not self.leave_as_float and self.original_dtype != array.dtype:
            if numpy.issubdtype(self.original_dtype, numpy.integer):
                # If we cast back to integer, we need to avoid overflows first!
                type_info = numpy.iinfo(self.original_dtype)
                array = array.clip(type_info.min, type_info.max, out=array)
            array = array.astype(self.original_dtype)

        return array
