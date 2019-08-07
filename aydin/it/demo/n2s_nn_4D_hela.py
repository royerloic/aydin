import time

import napari
import numpy as np
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor


def demo():

    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)
    image = image[0:10, 15:35, 130:167, 130:177].astype(np.float32)
    print(image.shape)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')

        generator = FastMultiscaleConvolutionalFeatures(max_level=3)

        regressor = NNRegressor(max_epochs=5)

        it = ImageTranslatorClassic(generator, regressor, normaliser_type='identity')

        start = time.time()
        it.train(image, image)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer.add_image(denoised, name='denoised')


demo()
