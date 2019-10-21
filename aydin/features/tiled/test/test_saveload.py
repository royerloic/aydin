import time
from os.path import join

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.features.base import FeatureGeneratorBase
from aydin.features.tiled.tiled_features import TiledFeatureGenerator
from aydin.io.folders import get_temp_folder


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_fg_saveload():

    scales = [1, 5, 7]
    widths = [5, 5, 5]
    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_scale_one=False,
    )

    generator_tiled = TiledFeatureGenerator(generator, max_tile_size=128)

    temp_file = join(get_temp_folder(), "test_fg_saveload.json" + str(time.time()))
    generator_tiled.save(temp_file)

    del generator_tiled

    loaded_generator = FeatureGeneratorBase.load(temp_file)
    assert len(loaded_generator.feature_generator.kernel_widths) == 3
    assert loaded_generator.feature_generator.kernel_scales[1] == 5

    image = n(camera().astype(numpy.float32))
    features = loaded_generator.compute(image, exclude_center_feature=True)
    assert features is not None

    for (
        feature_description
    ) in loaded_generator.feature_generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(loaded_generator.feature_generator.debug_feature_description_list) == 38
