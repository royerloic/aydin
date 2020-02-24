import time
import pytest
from os.path import join

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.folders import get_temp_folder
from aydin.it.it_base import ImageTranslatorBase
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor

from aydin.it.it_cnn import ImageTranslatorCNN


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_saveload_1():
    saveload('percentile', FastMultiscaleConvolutionalFeatures(), GBMRegressor())


@pytest.mark.heavy
def test_saveload_2():
    saveload(
        'minmax', FastMultiscaleConvolutionalFeatures(), NNRegressor(max_epochs=10)
    )


# @pytest.mark.heavy
def test_saveload_3():
    saveload(
        'minmax',
        FastMultiscaleConvolutionalFeatures(),
        GBMRegressor(),
        batch_size=100 * 1e3,
    )


def saveload(normaliser_type, generator, regressor, batch_size=None):

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

    assert psnr_denoised > 10 and ssim_denoised > 0.80


def saveload_cnn():
    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = numpy.expand_dims(
        numpy.expand_dims(noisy.astype(numpy.float32), axis=-1), axis=0
    )
    imageGT = numpy.expand_dims(
        numpy.expand_dims(image.astype(numpy.float32), axis=-1), axis=0
    )

    it = ImageTranslatorCNN()
    it.train(noisy, imageGT, supervised=False, shiftconv=True, max_epochs=1)

    # values before loading
    denoised = it.translate(noisy)
    psnr_denoised0 = psnr(denoised.squeeze(), image.squeeze())
    ssim_denoised0 = ssim(denoised.squeeze(), image.squeeze())
    print("noisy", psnr_denoised0, ssim_denoised0)

    temp_file = join(get_temp_folder(), "test_it_saveload_cnn" + str(time.time()))
    print(f"savepath: {temp_file}")
    it.save(temp_file)
    del it

    loaded_it = ImageTranslatorBase.load(temp_file)

    denoised = loaded_it.translate(noisy)
    psnr_denoised1 = psnr(denoised.squeeze(), image)
    ssim_denoised1 = ssim(denoised.squeeze(), image)
    print("denoised", psnr_denoised1, ssim_denoised1)

    # psnr_denoised = psnr(denoised, image)
    # ssim_denoised = ssim(denoised, image)
    # print("denoised", psnr_denoised, ssim_denoised)

    assert 0.99 < psnr_denoised0 / psnr_denoised1 < 1.01
    assert 0.99 < ssim_denoised0 / ssim_denoised1 < 1.01
