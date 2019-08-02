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
    def get_available_mem(self):
        """
        Returns available memory for feature generation
        :return: available memory in bytes
        :rtype: int
        """
        raise NotImplemented()

    @abstractmethod
    def get_needed_mem(self, num_elements):
        """
        Returns the amount of memory needed to generate features for an image of given size
        :param num_elements: number of elements (voxels)
        :type num_elements: int
        :return: memory needed in bytes
        :rtype: int
        """
        raise NotImplemented()

    @abstractmethod
    def is_enough_memory(self, num_elements):
        """
        Returns true if there is enough memory to generate features
        :param num_elements: number of elements (voxels)
        :type num_elements: int
        """
        raise NotImplemented()

    @abstractmethod
    def get_receptive_field_radius(self):
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
