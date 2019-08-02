import numpy

from aydin.normaliser.normaliser_base import NormaliserBase


class IdentityNormaliser(NormaliserBase):
    """
        Identity Normaliser

    """

    def __init__(self, clip=True, leave_as_float=False):
        """
        Constructs a normaliser
        """
        super().__init__(clip, leave_as_float)

    def calibrate(self, array):

        self.original_dtype = array.dtype

    def normalise(self, array: numpy.ndarray):
        """
        Normalises the given array in-place (if possible).

        :param array: array to normaliser
        :type array: ndarray
        """
        return super().normalise(array)

    def denormalise(self, array):
        """
        Denormalises the given array in-place (if possible).
        :param array: array to denormalise
        :type array: ndarray
        """

        return super().denormalise(array)
