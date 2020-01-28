import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.regression.clnn import CLNNRegressor
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def test_lgbm_regressor():
    regressor = GBMRegressor(n_estimators=600)
    with_regressor(regressor)


def test_nn_regressor():
    regressor = NNRegressor(max_epochs=40, depth=6)
    with_regressor(regressor)


def test_clnn_regressor():
    regressor = CLNNRegressor(max_epochs=40, depth=6)
    with_regressor(regressor)


def with_regressor(regressor):

    image = camera().astype(numpy.float32)
    image = n(image)

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = FastMultiscaleConvolutionalFeatures(exclude_scale_one=True)

    features = generator.compute(noisy, exclude_center_value=True)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    regressor.fit(x, y, x_valid=x, y_valid=y)

    yp = regressor.predict(x)

    denoised = yp.reshape(image.shape)

    denoised = numpy.clip(denoised, 0, 1)

    ssim_value = ssim(denoised, image)
    psnr_value = psnr(denoised, image)

    print("denoised", psnr_value, ssim_value)

    assert ssim_value > 0.80
