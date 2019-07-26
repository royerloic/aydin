import time

import napari
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = image.astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()

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
            num_leaves=127,
            max_bin=512,
            n_estimators=2048,
            early_stopping_rounds=20,
        )

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser='identity'
        )

        size = 128
        monitoring_image = noisy[
            256 - size // 2 : 256 + size // 2, 256 - size // 2 : 256 + size // 2
        ]

        def callback(iter, eval_metric, images):
            print(f"Iteration: {iter} metric: {eval_metric}")
            print(f"images: {str(images)}")
            if images:
                viewer.add_image(
                    rescale_intensity(images[0], in_range='image', out_range=(0, 1)),
                    name='noisy',
                )
            pass

        start = time.time()
        denoised = it.train(
            noisy,
            noisy,
            callbacks=[callback],
            # monitoring_images=[monitoring_image]
        )

        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        if denoised is None:
            # in case of batching we have to do this:
            start = time.time()
            denoised = it.translate(noisy)
            stop = time.time()
            print(f"inference: elapsed time:  {stop-start} ")

        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )
        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )
        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo(camera())
