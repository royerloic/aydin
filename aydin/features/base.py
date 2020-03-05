import math
import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle
import numpy
import psutil

from aydin.util.json import encode_indent
from aydin.util.log.log import lprint, lsection
from aydin.util.offcore.offcore import offcore_array


class FeatureGeneratorBase(ABC):
    """
        Feature Generator base class

    """

    def __init__(self):
        """
        Constructs a feature generator
        """

        self.check_nans = False
        self.debug_force_memmap = False

        # Impelmentations must initialise the dtype so that feature arrays can be created with correct type:
        self.dtype = None

    def save(self, path: str):
        """
        Saves a 'all-batteries-inlcuded' feature generator at a given path (folder).
        :param path: path to save to
        """
        os.makedirs(path, exist_ok=True)

        frozen = encode_indent(self)

        lprint(f"Saving feature generator to: {path}")
        with open(join(path, "feature_generation.json"), "w") as json_file:
            json_file.write(frozen)

        return frozen

    @staticmethod
    def load(path: str):
        """
        Returns a 'all-batteries-inlcuded' feature generator from a given path (folder).
        :param model_path: path to load from.
        """

        lprint(f"Loading feature generator from: {path}")
        with open(join(path, "feature_generation.json"), "r") as json_file:
            frozen = json_file.read()

        thawed = jsonpickle.decode(frozen)

        thawed._load_internals(path)

        return thawed

    @abstractmethod
    def _load_internals(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def get_receptive_field_radius(self, nb_dim):
        """
        Returns the receptive field radius in pixels.
        :return: receptive field radius in pixels.
        :rtype: int
        """
        raise NotImplementedError()

    @abstractmethod
    def max_non_batch_dims(self):
        """
        Returns the maximum number of non-batch dimensions that this generator supports.
        :return: max non-batch dimensions.
        :rtype: int
        """
        raise NotImplementedError()

    def max_voxels(self):
        """
        Returns the maximum number of voxels that this generator supports.
        :return: maximum number of voxels.
        :rtype: int
        """
        return math.inf

    @abstractmethod
    def compute(
        self,
        image,
        exclude_center_feature=False,
        exclude_center_value=False,
        batch_dims=None,
        features=None,
        feature_last_dim=True,
    ):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (d,h,w,n) where n is the number of features.

        :param image: image for which features are computed
        :type image: ndarray
        :param exclude_center_feature:
        :type exclude_center_feature:
        :param batch_dims:
        :type batch_dims:
        :param exclude_center_value:
        :type exclude_center_value:
        :return: feature array
        :rtype: ndarray
        """

        raise NotImplementedError()

    def create_feature_array(self, image, nb_features):
        """
        Creates a feature array of the right size and possibly in a 'lazy' way using memory mapping.

        :param image: image for which features are created
        :type image:
        :param nb_features: number of features needed
        :type nb_features:
        :return: feature array
        :rtype:
        """

        with lsection(f'Creating feature array for image of shape: {image.shape}'):

            size_in_bytes = nb_features * image.size * numpy.dtype(self.dtype).itemsize
            free_mem_in_bytes = psutil.virtual_memory().free
            lprint(f'There is {int(free_mem_in_bytes / 1E6)} MB of free memory')
            lprint(f'Feature array will be {(size_in_bytes / 1E6)} MB.')

            # We take the heuristic that we need 20% extra memory available to be confortable:
            is_enough_memory = 1.2 * size_in_bytes < free_mem_in_bytes

            # That's the shape we need:
            shape = (nb_features,) + image.shape

            if not self.debug_force_memmap and is_enough_memory:
                lprint(
                    f'There is enough memory -- we do not need to use a mem mapped array.'
                )
                array = numpy.zeros(shape, dtype=self.dtype)

            else:
                lprint(f'There is not enough memory -- we will use a mem mapped array.')
                array = offcore_array(shape=shape, dtype=self.dtype)

            return array
