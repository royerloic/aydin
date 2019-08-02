import dask
import numpy

from aydin.normaliser.normaliser_base import NormaliserBase


class PercentileNormaliser(NormaliserBase):
    """
        Percentile Normaliser

    """

    percent: float

    def __init__(
        self, percent: float = 0.001, clip=True, leave_as_float=False, epsilon=1e-21
    ):
        """
        Constructs a normaliser
        """

        super().__init__(clip, leave_as_float, epsilon)

        self.percent = percent

    def calibrate(self, array):

        self.original_dtype = array.dtype

        p = self.percent

        if hasattr(array, '__dask_keys__'):
            self.rmin = dask.array.percentile(array.flatten(), 100 * p).compute()
            self.rmax = dask.array.percentile(array.flatten(), 100 - 100 * p).compute()
        else:
            self.rmin = numpy.percentile(array, 100 * p)
            self.rmax = numpy.percentile(array, 100 - 100 * p)
