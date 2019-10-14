import math

import numpy
import numpy as np
import psutil
import pyopencl as cl
import scipy
from pyopencl.array import to_device, Array

from aydin.features.fast.features_1d import collect_feature_1d
from aydin.features.fast.features_2d import collect_feature_2d
from aydin.features.fast.features_3d import collect_feature_3d
from aydin.features.fast.features_4d import collect_feature_4d
from aydin.features.fast.integral import (
    integral_1d,
    integral_2d,
    integral_3d,
    integral_4d,
)
from aydin.features.features_base import FeatureGeneratorBase
from aydin.providers.opencl.opencl_provider import OpenCLProvider
from aydin.util.log.logging import lsection, lprint
from aydin.util.nd import nd_range_radii, nd_split_slices, remove_margin_slice
from aydin.util.offcore.offcore import offcore_array


class TiledFeatureGenerator(FeatureGeneratorBase):
    """
    Tiled feature generator.
    Takes an existing feature generator and applies it in a tiled fashion.

    """

    def __init__(self, feature_generator: FeatureGeneratorBase, max_tile_size=math.inf):
        """
        Constructs a tiled features generator given a delegated feature generator.

        :param feature_generator:
        :type FeatureGeneratorBase:

        """

        super().__init__()
        self.max_tile_size = max_tile_size
        self.feature_generator = feature_generator
        self.dtype = self.feature_generator.dtype

    def _load_internals(self, path: str):

        # We can't use the loaded provider,
        # we need a new one because of the
        # native ressources associated:
        self.opencl_provider = None

    def save(self, path: str):
        return super().save(path)

    @staticmethod
    def load(path: str):
        return super().load(path)

    def get_receptive_field_radius(self, nb_dim):
        return self.feature_generator.get_receptive_field_radius(nb_dim)

    def max_non_batch_dims(self):
        return self.feature_generator.max_non_batch_dims()

    def compute(
        self,
        image,
        batch_dims=None,
        exclude_center_feature=False,
        exclude_center_value=False,
        features=None,
        feature_last_dim=True,
    ):

        with lsection(
            f'Computing tiled features using feature generator: {type(self.feature_generator).__name__} '
        ):

            # Shape of  input image:
            shape = image.shape

            # set default batch_dim value:
            if batch_dims is None:
                batch_dims = (False,) * len(shape)

            # Number of non-batch dimensions:
            nb_non_batch_dims = sum((0 if is_batch else 1) for is_batch in batch_dims)
            lprint(f'Number of non-batch dimensions: {nb_non_batch_dims}')

            # Verify that we don't exceed the number of non-batch dimensions supported:
            assert nb_non_batch_dims <= self.max_non_batch_dims()
            assert nb_non_batch_dims > 0

            # Maximum number of voxels supported by delegated generator:
            max_voxels = self.feature_generator.max_voxels()
            lprint(f'Max voxels: {max_voxels}')

            # Based on the number of non-batch dimensions we compute the ideal tile size dimensions:
            self.tile_size = min(
                self.max_tile_size,
                int(round(math.pow(max_voxels, (1.0 / nb_non_batch_dims)))),
            )
            self.tile_size = max(1, self.tile_size)
            lprint(f'Tile size: {self.tile_size}')

            # This is the tile strategy, essentially how to split each dimension...
            tile_strategy = tuple(
                (1 if is_batch else int(math.ceil(dim / self.tile_size)))
                for dim, is_batch in zip(shape, batch_dims)
            )
            lprint(f'Tile strategy: {tile_strategy}')

            # Receptive field is:
            receptive_field_radius = self.get_receptive_field_radius(len(shape))
            lprint(f'Receptive field radius: {receptive_field_radius}')

            # How large the margins need to be to account for the receptive field:
            margins = (receptive_field_radius,) * len(shape)

            # We only need margins if we split a dimension:
            margins = tuple(
                (0 if split == 1 else margin)
                for margin, split in zip(margins, tile_strategy)
            )
            lprint(
                f'Tile margins: {margins}, receptive field: {self.get_receptive_field_radius(len(shape))}.'
            )

            # We compute the slices objects to cut the input and target images into batches:
            tile_slices = list(nd_split_slices(shape, tile_strategy, margins=margins))
            tile_slices_no_margins = list(nd_split_slices(shape, tile_strategy))

            # Zipping together slices with and without margins:
            slicezip = zip(tile_slices, tile_slices_no_margins)

            # Number of tiles:
            number_of_tiles = len(tile_slices)

            # For each tile we compute the features and assemble in the final feature array (use Dask/Zarr?):
            lprint(f'Number of tiles for computing features: {len(tile_slices)}')

            counter = 1
            for tile_slice, tile_slice_no_margins in slicezip:
                with lsection(
                    f'Computing features for tile {counter}/{number_of_tiles} '
                ):
                    lprint(f'Tile slice: {tile_slice}')

                    image_tile = image[tile_slice]
                    lprint(f'Tile shape: {image_tile.shape}')

                    features_tile = self.feature_generator.compute(
                        image_tile,
                        exclude_center_feature=exclude_center_feature,
                        exclude_center_value=exclude_center_value,
                        batch_dims=batch_dims,
                        features=None,  # we don
                        feature_last_dim=False,
                    )

                    # Just-in-time allocation:
                    if features is None:
                        # First we figure out the number of features:
                        nb_features = features_tile.shape[0]

                        # Allocate feature array:
                        features = self.create_feature_array(image, nb_features)

                    # Get the slice that 'cuts' out the margin:
                    remove_margin_slice_tuple = remove_margin_slice(
                        shape, tile_slice, tile_slice_no_margins
                    )

                    # copy contents of feature tile to main feature array:
                    features[(slice(None),) + tile_slice_no_margins] = features_tile[
                        (slice(None),) + remove_margin_slice_tuple
                    ]

                    counter += 1

            # 'collect_fesatures_nD' puts the feature vector in axis 0.
            # The following line creates a view of the array
            # in which the features are indexed by the last dimension instead:
            if feature_last_dim:
                features = np.moveaxis(features, 0, -1)

            return features
