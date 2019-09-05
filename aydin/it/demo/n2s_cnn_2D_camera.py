import time

import napari
import numpy
import skimage
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise

from aydin.it.it_cnn import ImageTranslatorCNN
from aydin.util.log.logging import set_log_max_depth


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    set_log_max_depth(5)

    image = camera().astype(numpy.float32)
    image = n(image)

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    # Classical denoisers:
    median1 = skimage.filters.median(noisy, disk(1))
    median2 = skimage.filters.median(noisy, disk(2))
    median5 = skimage.filters.median(noisy, disk(5))
    nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    # CNN based Image translation:
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)
    # input_dim only includes H, W, C; number of images is not included
    it = ImageTranslatorCNN(input_dim=noisy.shape[1:], supervised=False, shiftconv=True)

    start = time.time()
    denoised = it.train(noisy, noisy)
    denoised = denoised.reshape(image.shape)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy)
    denoised_inf = denoised_inf.reshape(image.shape)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)

    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised    :", psnr(image, denoised), ssim(denoised, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(nlm), name='nlm')
        viewer.add_image(n(median1), name='median1')
        viewer.add_image(n(median2), name='median2')
        viewer.add_image(n(median5), name='median5')
        viewer.add_image(n(denoised), name='denoised')
        viewer.add_image(n(denoised_inf), name='denoised_inf')


demo()
