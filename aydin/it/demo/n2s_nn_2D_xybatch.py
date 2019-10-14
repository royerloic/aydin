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
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = n(camera().astype(np.float32))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')

        generator = FastMultiscaleConvolutionalFeatures()  # max_level=4)
        regressor = NNRegressor(max_epochs=1)

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser_type='identity'
        )

        start = time.time()
        it.train(noisy, noisy, patience=10, max_epochs=100)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        # in case of batching we have to do this:
        start = time.time()
        denoised = it.translate(noisy)
        stop = time.time()
        print(f"inference: elapsed time:  {stop-start} ")

        image = numpy.clip(image, 0, 1)
        noisy = numpy.clip(noisy, 0, 1)
        denoised = numpy.clip(denoised, 0, 1)

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(n(denoised), name='denoised')


demo()
