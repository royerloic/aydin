from abc import ABC, abstractmethod


class FeatureGeneratorBase(ABC):
    """
        Feature Generator base class

    """

    def __init__(self):
        """
        Constructs a feature generator
        """

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
