import time

import napari
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.nn.nn import NNRegressor

"""
    Demo for self-supervised denoising using camera image with synthetic noise
"""


def demo():

    image = camera().astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    # image, _ = io.imread(examples_single.fmdd_hv110.get_path())
    # image = image.astype(np.float32)
    #
    # image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    # noisy = image

    scales = [1, 3, 7, 15, 31, 63, 127, 255]
    widths = [3, 3, 3, 3, 3, 3, 3, 3]

    start_time = time.time()
    regressor = NNRegressor(loss='l1')
    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths, kernel_scales=scales
    )
    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser='identity'
    )
    denoised = it.train(noisy, noisy)

    elapsedtime = time.time() - start_time
    print(f"time elapsed: {elapsedtime} s")

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )
        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )


demo()
