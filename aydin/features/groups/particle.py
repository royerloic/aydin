import numpy
from numpy.linalg import norm
from scipy.fft import idstn
from scipy.ndimage import gaussian_filter

from aydin.features.groups.correlation import CorrelationFeatures


class ParticleFeatures(CorrelationFeatures):
    """
    ParticleFeatures Feature Group class

    Generates features specialised for single diffraction limited molecules.
    """

    def __init__(
        self,
        size: int = 9,
        min_sigma: float = 0.75,
        max_sigma: float = 1.5,
        num_features: int = 8,
    ):
        """
        Constructor that configures these features.

        Parameters
        ----------
        size : int
            Size of the DCT filters

        min_sigma : float
            Minimum sigma of particle's gaussian.

        max_sigma : float
            Maximum sigma of particle's gaussian.

        num_features : int
            Number of features.


        """
        super().__init__(kernels=None)
        self.size = size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_features_ = num_features

        self.image = None
        self.exclude_center: bool = False

    def _ensure_particle_kernels_available(self, ndim: int):
        # Ensures that the kernels are available for subsequent steps.
        # We can't construct the kernels until we know the dimension of the image
        if self.kernels is None or self.kernels[0].ndim != ndim:
            kernels = []
            shape = tuple((self.size,) * ndim)

            sigmas = numpy.linspace(self.min_sigma, self.max_sigma, self.num_features_)

            for sigma in sigmas:
                kernel = numpy.zeros(shape=shape, dtype=numpy.float32)
                kernel.ravel()[kernel.size // 2] = 1.0
                kernel = gaussian_filter(kernel, sigma=sigma, truncate=8)
                kernel /= kernel.sum() + 1e-9
                kernels.append(kernel)

            self.kernels = kernels

    @property
    def receptive_field_radius(self) -> int:
        return self.size // 2

    def num_features(self, ndim: int) -> int:
        self._ensure_particle_kernels_available(ndim)
        return super().num_features(ndim)

    def prepare(self, image, excluded_voxels=None, **kwargs):
        if excluded_voxels is None:
            excluded_voxels = []

        self._ensure_particle_kernels_available(image.ndim)
        super().prepare(image, excluded_voxels, **kwargs)
