import time

import napari
import numpy as np
from skimage.data import astronaut
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def n(image):
    return rescale_intensity(
        image.astype(np.float32), in_range='image', out_range=(0, 1)
    )


def demo(image: np.ndarray):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    amplitude = 128
    noisy = image.astype(np.int16) + np.random.randint(
        -amplitude, amplitude, size=image.shape, dtype=np.int16
    )
    noisy = noisy.clip(0, 255).astype(np.uint8)

    # image = image.astype(np.float32)
    # noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(image), name='image', multichannel=True)
        viewer.add_image(n(noisy), name='noisy', multichannel=True)

        scales = [1, 3, 7, 15, 31, 63, 127, 255]
        widths = [3, 3, 3, 3, 3, 3, 3, 3]

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths,
            kernel_scales=scales,
            kernel_shapes=['l1'] * len(scales),
            exclude_center=True,
        )

        regressor = GBMRegressor(
            learning_rate=0.01,
            num_leaves=512,
            n_estimators=2048,
            early_stopping_rounds=20,
        )

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser='identity'
        )

        start = time.time()
        denoised = it.train(noisy, noisy)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        # start = time.time()
        # denoised = it.translate(noisy)
        # stop = time.time()
        # print(f"inference: elapsed time:  {stop-start} ")

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

        viewer.add_image(n(denoised), name='denoised', multichannel=True)


demo(astronaut())
