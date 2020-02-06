import time
import pytest
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.it.it_cnn import ImageTranslatorCNN


@pytest.mark.heavy
def test_it_cnn_shiftconv():
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = numpy.expand_dims(
        numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
    )
    it = ImageTranslatorCNN()

    start = time.time()
    it.train(noisy, noisy, supervised=False, shiftconv=True)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.squeeze(), 0, 1)
    denoised = numpy.clip(denoised.squeeze(), 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # PSNR and SSIM may be lower than those of classic methods.
    # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 23 and ssim_denoised > 0.81


@pytest.mark.heavy
def test_it_cnn_masking():
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = numpy.expand_dims(
        numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
    )
    it = ImageTranslatorCNN()

    start = time.time()
    it.train(noisy, noisy, supervised=False, shiftconv=False)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.squeeze(), 0, 1)
    denoised = numpy.clip(denoised.squeeze(), 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # PSNR and SSIM may be lower than those of classic methods.
    # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 23 and ssim_denoised > 0.81


@pytest.mark.heavy
def test_it_cnn_supervised():
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = numpy.expand_dims(
        numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
    )
    it = ImageTranslatorCNN()

    start = time.time()
    it.train(noisy, noisy, supervised=True, shiftconv=False)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.squeeze(), 0, 1)
    denoised = numpy.clip(denoised.squeeze(), 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy

    # PSNR and SSIM may be lower than those of classic methods.
    # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
    # do not change the number below, but instead, fix the problem -- most likely a parameter.

    assert psnr_denoised > 23 and ssim_denoised > 0.81
