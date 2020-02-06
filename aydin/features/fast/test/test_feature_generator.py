import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.util.log.log import Log


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_collect_feature_2d():

    scales = [1, 3, 7]
    widths = [3, 3, 3]

    Log.set_log_max_depth(2)

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l2'] * len(scales),
        exclude_scale_one=False,
    )

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image, exclude_center_feature=True)

    assert features is not None

    for feature_description in generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(generator.debug_feature_description_list) == 14


def test_collect_feature_2d_uint8():

    scales = [1, 3, 7]
    widths = [5, 5, 5]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        dtype=numpy.uint8,
    )

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image, exclude_center_feature=True)

    assert features is not None
    assert features.dtype == numpy.uint8


def test_collectscale_two_features_2d():

    scales = [2, 3, 7]
    widths = [1, 3, 3]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_scale_one=False,
    )

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image, exclude_center_feature=True)

    assert features is not None

    for feature_description in generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(generator.debug_feature_description_list) == 14
