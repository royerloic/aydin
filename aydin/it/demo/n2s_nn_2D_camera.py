import time

# import napari
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

"""
    Demo for self-supervised denoising using camera image with synthetic noise
"""


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():

    image = camera().astype(np.float32)
    image = n(image)
    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    start_time = time.time()

    generator = FastMultiscaleConvolutionalFeatures()
    regressor = NNRegressor()

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    it.train(noisy, noisy, max_epochs=30, patience=10)

    elapsedtime = time.time() - start_time
    print(f"time elapsed: {elapsedtime} s")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    print("noisy", psnr(noisy, image), ssim(noisy, image))
    print("denoised", psnr(denoised, image), ssim(denoised, image))

    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(n(image), name='image')
    #     viewer.add_image(n(noisy), name='noisy')
    #     viewer.add_image(n(denoised), name='denoised')


demo()
