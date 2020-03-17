# flake8: noqa
import time
from statistics import median

import numpy
from scipy.signal import convolve2d
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.io.datasets import normalise, add_noise, newyork, characters, examples_single
from aydin.it.it_skipnet import SkipNetImageTranslator
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def demo(image):

    # image = image[0:1024, 0:1024]

    image = normalise(image.astype(numpy.float32))

    noisy_image = add_noise(image)
    # noisy_image = image

    it = SkipNetImageTranslator(
        max_epochs=1000, learning_rate=0.01, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy_image, noisy_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    print(denoised_inf.shape)

    image = numpy.clip(image, 0, 1)
    noisy_image = numpy.clip(noisy_image, 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)
    print("noisy       :", psnr(image, noisy_image), ssim(noisy_image, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy_image), name='noisy')
        viewer.add_image(normalise(denoised_inf), name='denoised')

    return ssim(denoised_inf, image)


image = camera()
demo(image)
