import os
from abc import ABC, abstractmethod
from os.path import join

import jsonpickle

from aydin.util.json import encode_indent
from aydin.util.log.logging import lprint


class FeatureGeneratorBase(ABC):
    """
        Feature Generator base class

    """

    def __init__(self):
        """
        Constructs a feature generator
        """

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
    def is_enough_memory(self, array):
        """
        Returns true if there is enough memory to generate features
        :param array: image (voxels)
        """
        raise NotImplemented()

    @abstractmethod
    def get_receptive_field_radius(self, nb_dim):
        """
        Returns the receptive field radius in pixels.
        :return: receptive field radius in pixels.
        :rtype: int
        """
        raise NotImplemented()

    @abstractmethod
    def compute(self, image, batch_dims=None, features=None):
        """
        Computes the features given an image. If the input image is of shape (d,h,w),
        resulting features are of shape (d,h,w,n) where n is the number of features.
        :param image: image for which features are computed
        :type image: ndarray
        :return: feature array
        :rtype: ndarray
        """
