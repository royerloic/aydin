import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_collect_feature_2d_():

    scales = [1, 3, 7]
    widths = [5, 5, 5]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_center=True,
    )

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image)

    assert features is not None

    for feature_description in generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(generator.debug_feature_description_list) == 32


def test_collect_feature_2d_uint8():

    scales = [1, 3, 7]
    widths = [5, 5, 5]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_center=True,
        dtype=numpy.uint8,
    )

    image = n(camera().astype(numpy.float32))

    features = generator.compute(image)

    assert features is not None
    assert features.dtype == numpy.uint8
