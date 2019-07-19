import time

import numpy
from napari.util import app_context
from skimage.exposure import rescale_intensity

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor
from pitl.features.classic.mcfocl import MultiscaleConvolutionalFeatures


def demo(image):
    from napari import Viewer

    with app_context():

        level = 2
        scales = [1, 3, 7, 15, 31]
        widths = [3, 3, 3, 3, 3]

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths[:level], kernel_scales=scales[:level]
        )

        regressor = GBMRegressor(
            num_leaves=128,
            n_estimators=128,
            learning_rate=0.01,
            loss='l1',
            early_stopping_rounds=None,
        )

        it = ImageTranslatorClassic(generator, regressor)

        batch_dims = (False, False, False, False)

        start = time.time()
        it.train(image, image, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer = Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')

        print(image.shape)
        print(denoised.shape)


# (3, 320, 865, 1014)
image_path = examples_single.gardner_org.get_path()
array, metadata = io.imread(image_path)
print(array.shape)
array = array[:, 0:60, 270:500, 400:600]
array = array.astype(numpy.float32)
array = rescale_intensity(array, in_range='image', out_range=(0, 1))
print(array.shape)
demo(array)
