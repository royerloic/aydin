import time

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import pollen, newyork, normalise, add_noise
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.linear import LinearRegressor


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')

        generator = FastMultiscaleConvolutionalFeatures()

        regressor = LinearRegressor(mode='huber')

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser_type='identity'
        )

        start = time.time()
        denoised = it.train(noisy, noisy)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        if denoised is None:
            # in case of batching we have to do this:
            start = time.time()
            denoised = it.translate(noisy)
            stop = time.time()
            print(f"inference: elapsed time:  {stop-start} ")

        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        image = numpy.clip(image, 0, 1)
        noisy = numpy.clip(noisy, 0, 1)
        denoised = numpy.clip(denoised, 0, 1)

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))

        viewer.add_image(normalise(denoised), name='denoised')


camera_image = camera()
demo(camera_image)
lizard_image = camera()
demo(lizard_image)
pollen_image = pollen()
demo(pollen_image)
newyork_image = newyork()
demo(newyork_image)
