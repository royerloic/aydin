import time


import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import normalise, add_noise, pollen, newyork, lizard
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.rf import RandomForrestRegressor


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    generator = FastMultiscaleConvolutionalFeatures()

    regressor = RandomForrestRegressor()

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    denoised = it.translate(noisy)

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
