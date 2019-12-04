import time
from os.path import join, exists

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.folders import get_temp_folder
from aydin.providers.plaidml.plaidml_provider import PlaidMLProvider
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor
from aydin.regression.regressor_base import RegressorBase


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_gbm_saveload():
    saveload(GBMRegressor(n_estimators=128), min_ssim=0.78)


def test_nn_saveload():
    saveload(NNRegressor(max_epochs=5), min_ssim=0.78)


def saveload(regressor, min_ssim=0.80):
    provider = PlaidMLProvider()  # Needed to run in pytest

    image = camera().astype(numpy.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = FastMultiscaleConvolutionalFeatures()

    features = generator.compute(noisy, exclude_center_value=True)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    regressor.fit(x, y)

    temp_file = join(get_temp_folder(), "test_reg_saveload.json" + str(time.time()))
    regressor.save(temp_file)

    del regressor

    loaded_regressor = RegressorBase.load(temp_file)

    yp = loaded_regressor.predict(x)

    denoised = yp.reshape(image.shape)

    ssim_value = ssim(denoised, image)
    psnr_value = psnr(denoised, image)

    print("denoised", psnr_value, ssim_value)

    assert ssim_value > min_ssim
