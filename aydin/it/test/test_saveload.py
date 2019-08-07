import time
from os import remove
from os.path import join, exists

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io.folders import get_temp_folder
from aydin.it.it_base import ImageTranslatorBase
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.normaliser.minmax import MinMaxNormaliser
from aydin.normaliser.normaliser_base import NormaliserBase
from aydin.normaliser.percentile import PercentileNormaliser
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_saveload_1():
    saveload('percentile', FastMultiscaleConvolutionalFeatures(), GBMRegressor())


def test_saveload_2():
    saveload(
        'minmax', FastMultiscaleConvolutionalFeatures(), NNRegressor(max_epochs=10)
    )


def saveload(normaliser_type, generator, regressor):

    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    it = ImageTranslatorClassic(
        normaliser_type=normaliser_type,
        feature_generator=generator,
        regressor=regressor,
    )

    it.train(noisy, noisy)

    temp_file = join(get_temp_folder(), "test_it_saveload" + str(time.time()))
    it.save(temp_file)
    del it

    loaded_it = ImageTranslatorBase.load(temp_file)

    denoised = loaded_it.translate(noisy)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressohave   been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 24 and ssim_denoised > 0.80
