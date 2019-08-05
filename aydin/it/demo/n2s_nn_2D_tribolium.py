import time
from os.path import join

import napari
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from tifffile import imread

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import examples_zipped
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn.nn import NNRegressor


def demo():
    """
        Demo for supervised denoising using CARE's tribolium example as a montage.

    """

    image = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_GT_montage.tif'
        )
    ).astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    noisy = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_train_low_montage.tif'
        )
    ).astype(np.float32)
    noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))

    image_test = imread(
        join(examples_zipped.care_tribolium.get_path(), 'tribolium_test_GT_montage.tif')
    ).astype(np.float32)
    image_test = rescale_intensity(image_test, in_range='image', out_range=(0, 1))

    noisy_test = imread(
        join(
            examples_zipped.care_tribolium.get_path(), 'tribolium_test_low_montage.tif'
        )
    ).astype(np.float32)
    noisy_test = rescale_intensity(noisy_test, in_range='image', out_range=(0, 1))

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )

        scales = [1, 3, 7, 15, 31]
        widths = [3, 3, 3, 3, 3]

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths, kernel_scales=scales
        )

        regressor = NNRegressor(max_epochs=5)

        it = ImageTranslatorClassic(
            feature_generator=generator, regressor=regressor, normaliser='identity'
        )

        start = time.time()
        denoised = it.train(noisy, image)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised_test = it.translate(noisy_test)
        stop = time.time()
        print(f"inference: elapsed time:  {stop-start} ")

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        print(
            "denoised_test",
            psnr(denoised_test, image_test),
            ssim(denoised_test, image_test),
        )

        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )
        viewer.add_image(
            rescale_intensity(image_test, in_range='image', out_range=(0, 1)),
            name='image_test',
        )
        viewer.add_image(
            rescale_intensity(noisy_test, in_range='image', out_range=(0, 1)),
            name='noisy_test',
        )
        viewer.add_image(
            rescale_intensity(denoised_test, in_range='image', out_range=(0, 1)),
            name='denoised_test',
        )


demo()
