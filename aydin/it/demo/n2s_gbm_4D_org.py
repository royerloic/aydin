import time

import napari
import numpy
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def demo():

    with napari.gui_qt():

        # (3, 320, 865, 1014)
        image_path = examples_single.gardner_org.get_path()
        image, metadata = io.imread(image_path, zarr_cache=False)
        print(image.shape)
        image = image.squeeze()
        image = image[:, 0:60, 270:500, 400:600]
        image = image.astype(numpy.float32)
        image = rescale_intensity(image, in_range='image', out_range=(0, 1))
        print(image.shape)

        generator = FastMultiscaleConvolutionalFeatures(max_level=2)

        regressor = GBMRegressor()

        it = ImageTranslatorClassic(generator, regressor, normaliser_type='identity')

        batch_dims = (False, False, False, False)

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


demo()
