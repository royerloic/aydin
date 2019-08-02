import dask
import numpy

from aydin.normaliser.normaliser_base import NormaliserBase


class MinMaxNormaliser(NormaliserBase):
    """
        Min-Max Normaliser

    """

    def __init__(self, clip=True, leave_as_float=False):
        """
        Constructs a normaliser
        """
        super().__init__(clip, leave_as_float)

    def calibrate(self, array):

        self.original_dtype = array.dtype

        if hasattr(array, '__dask_keys__'):
            self.rmin = dask.array.min(array.flatten()).compute()
            self.rmax = dask.array.max(array.flatten()).compute()
        else:
            self.rmin = numpy.min(array)
            self.rmax = numpy.max(array)
