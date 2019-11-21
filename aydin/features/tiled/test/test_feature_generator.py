import pytest
import numpy
from skimage.data import camera, binary_blobs
from skimage.exposure import rescale_intensity

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.features.tiled.tiled_features import TiledFeatureGenerator


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_collect_feature_2d():

    scales = [1, 3, 7]
    widths = [3, 3, 3]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_scale_one=False,
    )

    tiled_generator = TiledFeatureGenerator(generator, max_tile_size=256)

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image, exclude_center_feature=True)
    features_tiled = tiled_generator.compute(image, exclude_center_feature=True)

    assert features is not None
    assert features_tiled is not None

    assert features.shape == features_tiled.shape
    assert features.dtype == features_tiled.dtype

    avg_abs_diff = numpy.sum(numpy.abs(features - features_tiled)) / features.size

    print(f'Average absolute difference: {avg_abs_diff}')

    assert avg_abs_diff < 0.001


def test_collect_feature_1d():
    collect_feature_nd(1, 512)


def test_collect_feature_2d():
    collect_feature_nd(2, 256)


def test_collect_feature_3d():
    collect_feature_nd(3, 64)


@pytest.mark.heavy
def test_collect_feature_4d():
    collect_feature_nd(4, 32)


def collect_feature_nd(n_dim, tile_size):

    scales = [1, 3, 7]
    widths = [3, 3, 3]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_scale_one=False,
    )

    tiled_generator = TiledFeatureGenerator(generator, max_tile_size=tile_size)

    image = binary_blobs(length=tile_size * 2, seed=1, n_dim=n_dim).astype(
        numpy.float32
    )
    image = n(image)

    features = generator.compute(image, exclude_center_feature=True)
    features_tiled = tiled_generator.compute(image, exclude_center_feature=True)

    assert features is not None
    assert features_tiled is not None

    assert features.shape == features_tiled.shape
    assert features.dtype == features_tiled.dtype

    avg_abs_diff = numpy.sum(numpy.abs(features - features_tiled)) / features.size

    print(f'Average absolute difference: {avg_abs_diff}')

    assert avg_abs_diff < 0.001
