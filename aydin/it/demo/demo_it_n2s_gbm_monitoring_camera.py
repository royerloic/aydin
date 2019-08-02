import time

import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.it.monitor import Monitor
from aydin.regression.gbm import GBMRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = image.astype(np.float32)
    image = n(image)

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()

        size = 128
        monitoring_image = noisy[
            256 - size // 2 : 256 + size // 2, 256 - size // 2 : 256 + size // 2
        ]

        def callback(arg):
            print(arg)
            image, eval_metric, iter = arg
            print(f"Iteration: {iter} metric: {eval_metric}")
            # print(f"images: {str(images)}")
            # print("image: ", image[0])
            if image[0] is not None:
                viewer.add_image(
                    rescale_intensity(image[0], in_range='image', out_range=(0, 1)),
                    name='noisy',
                )

        generator = FastMultiscaleConvolutionalFeatures(max_features=50)
        regressor = GBMRegressor()
        monitor = Monitor(callback)

        it = ImageTranslatorClassic(
            feature_generator=generator,
            regressor=regressor,
            normaliser='identity',
            monitor=monitor,
        )

        start = time.time()

        denoised = it.train(noisy, noisy, monitoring_images=[monitoring_image])

        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))

        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(denoised), name='denoised')


demo(camera())
