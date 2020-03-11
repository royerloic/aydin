# flake8: noqa
import time

import numpy
import skimage
from skimage.data import camera
from skimage.filters import gaussian
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import normalise, add_noise, lizard, pollen, newyork, characters
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.it.it_cnn import ImageTranslatorCNN
from aydin.it.it_inverting_pt import InvertingImageTranslator
from aydin.regression.gbm import GBMRegressor


def demo(image, max_epochs=10):

    image = image[0:512, 0:512]

    image = 1 - normalise(image.astype(numpy.float32))

    blurred_image = gaussian(
        image, sigma=(1.5, 1.5), truncate=3.5, multichannel=False, preserve_range=True
    )

    noisy_image = add_noise(blurred_image)

    it = InvertingImageTranslator(max_epochs=10)

    start = time.time()
    it.train(noisy_image, noisy_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = numpy.clip(image, 0, 1)
    noisy_image = numpy.clip(noisy_image.reshape(image.shape), 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)
    print("noisy       :", psnr(image, noisy_image), ssim(noisy_image, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy_image), name='noisy')
        viewer.add_image(normalise(blurred_image), name='blurred')
        viewer.add_image(normalise(denoised_inf), name='denoised')


characters_image = characters()
demo(characters_image)
