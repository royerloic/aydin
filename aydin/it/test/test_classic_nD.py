import time
import pytest
import numpy
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn import NNRegressor
from aydin.util.log.log import Log

display_for_debug = False


@pytest.mark.heavy
def test_it_classic_nn_2D():
    it_classic_nD(2, 512, numpy.s_[0:281, 0:413], regressor='nn', min_ssim=0.60)


@pytest.mark.heavy
def test_it_classic_nn_3D():
    it_classic_nD(3, 128, numpy.s_[0:111, 0:113, 0:97], regressor='nn', min_ssim=0.70)


@pytest.mark.heavy
def test_it_classic_nn_4D():
    it_classic_nD(
        4, 64, numpy.s_[0:11, 0:41, 0:57, 0:53], regressor='nn', min_ssim=0.70
    )


@pytest.mark.heavy
def test_it_classic_gbm_2D():
    it_classic_nD(2, 512, numpy.s_[0:201, 0:213], regressor='gbm', min_ssim=0.60)


@pytest.mark.heavy
def test_it_classic_gbm_3D():
    it_classic_nD(3, 48, numpy.s_[0:41, 0:43, 0:37], regressor='nn', min_ssim=0.70)


@pytest.mark.heavy
def test_it_classic_gbm_4D():
    it_classic_nD(
        4, 48, numpy.s_[0:11, 0:23, 0:22, 0:21], regressor='nn', min_ssim=0.70
    )


@pytest.mark.heavy
def test_it_classic_gbm_2D_batchdims():
    Log.set_log_max_depth(2)
    it_classic_nD(
        2,
        256,
        numpy.s_[0:117, 0:175],
        regressor='gbm',
        batch_dims=(False, True),
        min_ssim=0.30,
    )


@pytest.mark.heavy
def test_it_classic_gbm_3D_batchdims():
    Log.set_log_max_depth(2)
    it_classic_nD(
        3,
        48,
        numpy.s_[0:31, 0:37, 0:41],
        regressor='gbm',
        batch_dims=(False, True, False),
        min_ssim=0.80,
    )


@pytest.mark.heavy
def test_it_classic_gbm_4D_batchdims():
    Log.set_log_max_depth(2)
    it_classic_nD(
        4,
        48,
        numpy.s_[0:11, 0:13, 0:17, 0:15],
        regressor='gbm',
        batch_dims=(False, True, False, True),
        min_ssim=0.66,
    )


def it_classic_nD(
    n_dim,
    length=128,
    train_slice=numpy.s_[...],
    regressor='nn',
    batch_dims=None,
    min_ssim=0.85,
):
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = binary_blobs(length=length, seed=1, n_dim=n_dim).astype(numpy.float32)
    image = n(image)

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.1, seed=0)
    noisy = noisy.astype(numpy.float32)

    train = noisy[train_slice]

    generator = FastMultiscaleConvolutionalFeatures()

    if regressor == 'nn':
        regressor = NNRegressor(max_epochs=10)
    elif regressor == 'gbm':
        regressor = GBMRegressor()

    it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(train, train, batch_dims=batch_dims)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised = it.translate(noisy, batch_dims=batch_dims)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    # if the line below fails, then the parameters of the image the lgbm regressor have been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert ssim_denoised > min_ssim

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )
