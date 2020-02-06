import time

import napari
import numpy
import numpy as np
import skimage
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.features.tiled.tiled_features import TiledFeatureGenerator
from aydin.io.datasets import newyork
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.util.log.log import Log


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    Log.set_log_max_depth(5)

    image = camera().astype(np.float32)  # newyork()[256:-256, 256:-256]
    image = n(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    median1 = skimage.filters.median(noisy, disk(1))
    median2 = skimage.filters.median(noisy, disk(2))
    median5 = skimage.filters.median(noisy, disk(5))

    nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    generator = FastMultiscaleConvolutionalFeatures(max_level=10)
    regressor = GBMRegressor()

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised = it.translate(noisy, tile_size=256)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised    :", psnr(image, denoised), ssim(denoised, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(nlm), name='nlm')
        viewer.add_image(n(median1), name='median1')
        viewer.add_image(n(median2), name='median2')
        viewer.add_image(n(median5), name='median5')
        viewer.add_image(n(denoised), name='denoised')


demo()
