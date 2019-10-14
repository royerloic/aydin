import time

import napari
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def demo(image, noisy):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = camera()

    intensity = 5
    numpy.random.seed(0)
    noisy = image.astype(numpy.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))
    noisy = numpy.random.poisson(noisy * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy *= 255
    noisy = noisy.astype(numpy.uint16)

    # Both image and noisy are uint32 within the [0, 255] range...
    # Builtin normalisation should do the job...

    generator = FastMultiscaleConvolutionalFeatures()
    regressor = GBMRegressor()

    it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = image.astype(numpy.float32)
    noisy = noisy.astype(numpy.float32)
    denoised = denoised.astype(numpy.float32)

    image = rescale_intensity(image, in_range='image', out_range=(0, 1))
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))
    denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

    print("noisy", psnr(image, noisy), ssim(noisy, image))
    print("denoised", psnr(image, denoised), ssim(denoised, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')


demo()
