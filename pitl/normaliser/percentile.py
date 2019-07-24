import dask
import numpy

from pitl.normaliser.normaliser_base import NormaliserBase


class PercentileNormaliser(NormaliserBase):
    """
        Percentile Normaliser

    """

    percent: int

    def __init__(
        self, percent: int = 1, clip=True, leave_as_float=False, epsilon=1e-21
    ):
        """
        Constructs a normaliser
        """

        super().__init__(clip, leave_as_float, epsilon)

        self.percent = percent

    def calibrate(self, array):

        self.original_dtype = array.dtype

        if hasattr(array, '__dask_keys__'):
            self.rmin = dask.array.percentile(array.flatten(), self.percent).compute()
            self.rmax = dask.array.percentile(
                array.flatten(), 100 - self.percent
            ).compute()
        else:
            self.rmin = numpy.percentile(array, self.percent)
            self.rmax = numpy.percentile(array, 100 - self.percent)
