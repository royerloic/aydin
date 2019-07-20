import time

import numpy
import napari
from skimage.exposure import rescale_intensity

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor


def demo(image):

    with napari.gui_qt():

        level = 2
        scales = [1, 3, 7, 15, 31][:level]
        widths = [3, 3, 3, 3, 3][:level]

        batch_dims = (True, False, False, False)

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths, kernel_scales=scales
        )

        regressor = GBMRegressor(
            num_leaves=128,
            n_estimators=128,
            learning_rate=0.01,
            loss='l1',
            early_stopping_rounds=None,
        )

        it = ImageTranslatorClassic(generator, regressor)

        start = time.time()
        it.train(image, image, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')

        print(image.shape)
        print(denoised.shape)


# (3, 320, 865, 1014)
image_path = examples_single.gardner_org.get_path()
array, metadata = io.imread(image_path)
print(array.shape)
array = array[:, 0:60, 270:500, 400:600]
print(array.shape)
array = array.astype(numpy.float32)
array = rescale_intensity(array, in_range='image', out_range=(0, 1))
demo(array)
