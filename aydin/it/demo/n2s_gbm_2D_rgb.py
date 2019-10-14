import time

import napari
import numpy as np
from skimage.data import astronaut
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def n(image):
    return rescale_intensity(
        image.astype(np.float32), in_range='image', out_range=(0, 255)
    )


def demo():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = astronaut()

    amplitude = 128
    noisy = image.astype(np.int16) + np.random.randint(
        -amplitude, amplitude, size=image.shape, dtype=np.int16
    )
    noisy = noisy.clip(0, 255).astype(np.uint8)

    # image = image.astype(np.float32)
    # noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image', multichannel=True)
        viewer.add_image(noisy, name='noisy', multichannel=True)

        generator = FastMultiscaleConvolutionalFeatures()
        regressor = GBMRegressor()

        it = ImageTranslatorClassic(
            feature_generator=generator,
            regressor=regressor,
            normaliser_type='percentile',
        )

        start = time.time()
        it.train(noisy, noisy)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(noisy)
        stop = time.time()
        print(f"Inference: elapsed time:  {stop-start} ")

        print(
            "noisy",
            psnr(n(image), n(noisy), data_range=255),
            ssim(n(noisy), n(image), multichannel=True),
        )
        print(
            "denoised",
            psnr(n(image), n(denoised), data_range=255),
            ssim(n(denoised), n(image), multichannel=True),
        )
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(denoised, name='denoised', multichannel=True)


demo()
