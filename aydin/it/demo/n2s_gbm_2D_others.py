import time

import napari
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )

        generator = FastMultiscaleConvolutionalFeatures()
        regressor = GBMRegressor()

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser_type='identity'
        )

        start = time.time()
        denoised = it.train(noisy, noisy)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        # start = time.time()
        # denoised = it.translate(noisy)
        # stop = time.time()
        # print(f"inference: elapsed time:  {stop-start} ")

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )


array, metadata = io.imread(examples_single.generic_crowd.get_path())
demo(array.astype(np.float32))

array, metadata = io.imread(examples_single.generic_mandrill.get_path())
demo(array.astype(np.float32))
