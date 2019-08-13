import time
from os import remove
from os.path import join, exists

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.features.features_base import FeatureGeneratorBase
from aydin.io.folders import get_temp_folder


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_fg_saveload():

    scales = [1, 5, 7]
    widths = [5, 5, 5]
    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths, kernel_scales=scales, kernel_shapes=['l1'] * len(scales)
    )

    temp_file = join(get_temp_folder(), "test_fg_saveload.json" + str(time.time()))
    generator.save(temp_file)

    del generator

    loaded_generator = FeatureGeneratorBase.load(temp_file)
    assert len(loaded_generator.kernel_widths) == 3
    assert loaded_generator.kernel_scales[1] == 5

    image = n(camera().astype(numpy.float32))
    features = loaded_generator.compute(image, exclude_center_feature=True)
    assert features is not None

    for feature_description in loaded_generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(loaded_generator.debug_feature_description_list) == 38
