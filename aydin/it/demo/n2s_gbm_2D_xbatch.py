import time

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = camera().astype(np.float32)
    image = n(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')

        generator = FastMultiscaleConvolutionalFeatures(max_level=10)
        regressor = GBMRegressor()

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser_type='identity'
        )

        batch_dims = (True, False)

        start = time.time()
        it.train(noisy, noisy, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(noisy, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference: elapsed time:  {stop-start} ")

        image = numpy.clip(image, 0, 1)
        noisy = numpy.clip(noisy, 0, 1)
        denoised = numpy.clip(denoised, 0, 1)

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(n(denoised), name='denoised')
        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo()
