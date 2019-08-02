import time

import napari
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def demo(image, noisy):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

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

        it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

        start = time.time()
        denoised = it.train(noisy, noisy)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        if denoised is None:
            # in case of batching we have to do this:
            start = time.time()
            denoised = it.translate(noisy)
            stop = time.time()
            print(f"inference: elapsed time:  {stop-start} ")

        viewer.add_image(image, name='image')
        viewer.add_image(noisy, name='noisy')
        viewer.add_image(denoised, name='denoised')

        image = image.astype(numpy.float32)
        noisy = noisy.astype(numpy.float32)
        denoised = denoised.astype(numpy.float32)

        image = rescale_intensity(image, in_range='image', out_range=(0, 1))
        noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))
        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


image = camera()

intensity = 5
numpy.random.seed(0)
noisy = image.astype(numpy.float32)
noisy = rescale_intensity(noisy, in_range='image', out_range=(0, 1))
noisy = numpy.random.poisson(noisy * intensity) / intensity
noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
noisy *= 255
noisy = noisy.astype(numpy.uint16)


# Both image and noisy are uint32 within the [0, 255] range...
# Builtin normalisation should do the job...

demo(image, noisy)
