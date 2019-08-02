import time

import napari
import numpy
from skimage.data import camera, binary_blobs
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor
from pitl.regression.nn.nn import NNRegressor


# Turns on napari display...
display_for_debug = False


def test_it_classic_nn_2D():
    it_classic_nn_nD(2, 512, numpy.s_[0:501, 0:373], regressor='nn')


def test_it_classic_nn_3D():
    it_classic_nn_nD(3, 128, numpy.s_[0:111, 0:73, 0:37], regressor='nn')


def test_it_classic_nn_4D():
    it_classic_nn_nD(4, 64, numpy.s_[0:11, 0:31, 0:3747, 0:50], regressor='nn')


def test_it_classic_gbm_2D():
    it_classic_nn_nD(2, 512, numpy.s_[0:111, 0:73], regressor='gbm')


def test_it_classic_gbm_3D():
    it_classic_nn_nD(3, 64, numpy.s_[0:111, 0:73, 0:37], regressor='gbm')


def test_it_classic_gbm_4D():
    it_classic_nn_nD(
        4, 32, numpy.s_[0:11, 0:31, 0:37, 0:50], regressor='gbm', min_ssim=0.70
    )


def test_it_classic_gbm_2D_batchdims():
    it_classic_nn_nD(
        2,
        256,
        numpy.s_[0:117, 0:175],
        regressor='gbm',
        batch_dims=(False, True),
        min_ssim=0.30,
    )


def test_it_classic_gbm_3D_batchdims():
    it_classic_nn_nD(
        3,
        64,
        numpy.s_[0:31, 0:37, 0:41],
        regressor='gbm',
        batch_dims=(False, True, False),
        min_ssim=0.80,
    )


def test_it_classic_gbm_4D_batchdims():
    it_classic_nn_nD(
        4,
        32,
        numpy.s_[0:11, 0:13, 0:17, 0:15],
        regressor='gbm',
        batch_dims=(False, True, False, True),
        min_ssim=0.66,
    )


def it_classic_nn_nD(
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

    generator = FastMultiscaleConvolutionalFeatures(max_features=10)

    if regressor == 'nn':
        regressor = NNRegressor()
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

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(n(train), name='train')
        viewer.add_image(n(image), name='image')
        viewer.add_image(n(noisy), name='noisy')
        viewer.add_image(n(denoised), name='denoised')

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # if the line below fails, then the parameters of the image the lgbm regressor have been broken.
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert ssim_denoised > min_ssim


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )
