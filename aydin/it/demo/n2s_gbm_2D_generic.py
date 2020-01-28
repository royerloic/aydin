import time

import numpy
import numpy as np
import skimage
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import newyork, pollen, normalise, add_noise, lizard
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.util.log.log import Log


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    # Log.set_log_max_depth(5)

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    median1 = skimage.filters.median(noisy, disk(1))
    median2 = skimage.filters.median(noisy, disk(2))
    median5 = skimage.filters.median(noisy, disk(5))

    nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    generator = FastMultiscaleConvolutionalFeatures()
    regressor = GBMRegressor(patience=20)

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy, noisy)
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

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised    :", psnr(image, denoised), ssim(denoised, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(nlm), name='nlm')
        viewer.add_image(normalise(median1), name='median1')
        viewer.add_image(normalise(median2), name='median2')
        viewer.add_image(normalise(median5), name='median5')
        viewer.add_image(normalise(denoised), name='denoised')


camera_image = camera()
demo(camera_image)
lizard_image = lizard()
demo(lizard_image)
pollen_image = pollen()
demo(pollen_image)
newyork_image = newyork()
demo(newyork_image)
