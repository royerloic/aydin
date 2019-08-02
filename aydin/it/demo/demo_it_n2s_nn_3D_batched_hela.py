import time

import napari
import numpy as np
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn.nn import NNRegressor


def demo(image):

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')

        scales = [1, 3, 7, 15, 31, 63]
        widths = [3, 3, 3, 3, 3, 3]

        batch_dims = (True, False, False, False)

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths, kernel_scales=scales, exclude_center=False
        )

        regressor = NNRegressor()

        it = ImageTranslatorClassic(generator, regressor, normaliser='identity')

        start = time.time()
        denoised = it.train(image, image, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        # start = time.time()
        # denoised = it.translate(image)
        # stop = time.time()
        # print(f"inference: elapsed time:  {stop-start} ")

        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )


image_path = examples_single.hyman_hela.get_path()
array, metadata = io.imread(image_path)
array = array[0:10, 15:35, 130:167, 130:177]
array = array.astype(np.float32)
array = rescale_intensity(array, in_range='image', out_range=(0, 1))
demo(array)
