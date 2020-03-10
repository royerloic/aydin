# flake8: noqa
import time

import napari
import numpy
from skimage.exposure import rescale_intensity

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor


def demo():

    image_path = examples_single.hyman_hela.get_path()
    image, metadata = io.imread(image_path)
    print(image.shape)
    image = image[0:10, 15:35, 130:167, 130:177].astype(numpy.float16)
    print(image.shape)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    generator = FastMultiscaleConvolutionalFeatures(max_level=7)

    regressor = NNRegressor()

    it = ImageTranslatorClassic(generator, regressor)

    start = time.time()
    it.train(image, image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    start = time.time()
    denoised = it.translate(image)
    stop = time.time()
    print(f"inference train: elapsed time:  {stop-start} ")

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')


demo()
