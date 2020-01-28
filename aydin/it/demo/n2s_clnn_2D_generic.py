import time

import numpy
import numpy as np
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.clnn.tensor.cltensor import CLTensor
from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import newyork, normalise, add_noise, pollen, lizard
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.clnn import CLNNRegressor

"""
    Demo for self-supervised denoising using camera image with synthetic noise
"""


def demo(image, max_epochs=50):

    image = normalise(image.astype(np.float32))
    noisy = add_noise(image)

    start_time = time.time()

    generator = FastMultiscaleConvolutionalFeatures()
    regressor = CLNNRegressor(max_epochs=max_epochs)

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    it.train(noisy, noisy)

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

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised), name='denoised')


camera_image = camera()
demo(camera_image)
lizard_image = lizard()
demo(lizard_image)
pollen_image = pollen()
demo(pollen_image)
newyork_image = newyork()
demo(newyork_image)
