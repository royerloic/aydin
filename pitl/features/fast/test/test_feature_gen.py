from pprint import pprint

import numpy
import pyopencl
from pyopencl.array import Array
from skimage.data import camera

from pitl.features.fast.features_1d import collect_feature_1d
from pitl.features.fast.features_2d import collect_feature_2d
from pitl.features.fast.features_3d import collect_feature_3d
from pitl.features.fast.features_4d import collect_feature_4d
from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.opencl.opencl_provider import OpenCLProvider


def test_collect_feature_2d():

    scales = [1, 3, 7]
    widths = [5, 5, 5]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths,
        kernel_scales=scales,
        kernel_shapes=['l1'] * len(scales),
        exclude_center=True,
    )

    image = camera().astype(numpy.float32)

    aspect_ratio = (0.5, 1)

    #  self, image, batch_dims=None, features=None, features_aspect_ratio=None
    generator.compute(image, features_aspect_ratio=aspect_ratio)

    for feature_description in generator.debug_feature_description_list:
        print(f"Feature: {feature_description}")

    assert len(generator.debug_feature_description_list) == 32
