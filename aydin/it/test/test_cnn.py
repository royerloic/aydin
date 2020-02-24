import time
import pytest
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise
from aydin.io.datasets import normalise, add_noise
from aydin.it.it_cnn import ImageTranslatorCNN


def test_it_cnn_shiftconv_light():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 1
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    it = ImageTranslatorCNN(
        training_architecture='shiftconv',
        num_layer=2,
        batch_norm=None,  # 'instance',
        max_epochs=max_epochs,
        verbose=1,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)
    denoised = denoised.reshape(image.shape)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop-start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    # assert psnr_denoised > 20 and ssim_denoised > 0.6


def test_it_cnn_checkerbox_light():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 1
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    it = ImageTranslatorCNN(
        training_architecture='checkerbox',
        num_layer=2,
        mask_shape=(3, 3),
        batch_norm='instance',
        max_epochs=max_epochs,
        verbose=1,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)
    denoised = denoised.reshape(image.shape)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop-start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    # assert psnr_denoised > 20 and ssim_denoised > 0.6


def test_it_cnn_random_light():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 4
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    it = ImageTranslatorCNN(
        training_architecture='random',
        num_layer=2,
        batch_norm='instance',
        max_epochs=max_epochs,
        verbose=1,
    )
    it.train(noisy, noisy)
    denoised = it.translate(noisy, tile_size=image_width)
    denoised = denoised.reshape(image.shape)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(noisy, image)
    ssim_noisy = ssim(noisy, image)
    print("noisy", psnr_noisy, ssim_noisy)

    psnr_denoised = psnr(denoised, image)
    ssim_denoised = ssim(denoised, image)
    print("denoised", psnr_denoised, ssim_denoised)

    stop = time.time()
    print(f"Total elapsed time: {stop-start} ")
    assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
    # assert psnr_denoised > 20 and ssim_denoised > 0.7


# @pytest.mark.heavy
# def test_it_cnn_shiftconv():
#     """
#         Test for self-supervised denoising using camera image with synthetic noise
#     """
#
#     image = rescale_intensity(
#         camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
#     )
#
#     intensity = 5
#     numpy.random.seed(0)
#     noisy = numpy.random.poisson(image * intensity) / intensity
#     noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
#     noisy = numpy.expand_dims(
#         numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
#     )
#     it = ImageTranslatorCNN(training_architecture=True)
#
#     start = time.time()
#     it.train(noisy, noisy)  # max_epochs=100
#     stop = time.time()
#     print(f"####### Training: elapsed time:  {stop-start} sec")
#
#     start = time.time()
#     denoised = it.translate(noisy)
#     stop = time.time()
#     print(f"####### Inference: elapsed time:  {stop-start} sec")
#
#     image = numpy.clip(image, 0, 1)
#     noisy = numpy.clip(noisy.squeeze(), 0, 1)
#     denoised = numpy.clip(denoised.squeeze(), 0, 1)
#
#     psnr_noisy = psnr(noisy, image)
#     ssim_noisy = ssim(noisy, image)
#     print("noisy", psnr_noisy, ssim_noisy)
#
#     psnr_denoised = psnr(denoised, image)
#     ssim_denoised = ssim(denoised, image)
#     print("denoised", psnr_denoised, ssim_denoised)
#
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#
#     # PSNR and SSIM may be lower than those of classic methods.
#     # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
#     # do not change the number below, but instead, fix the problem -- most likely a parameter.
#
#     assert psnr_denoised > 23 and ssim_denoised > 0.81
#
#
# @pytest.mark.heavy
# def test_it_cnn_masking():
#     """
#         Test for self-supervised denoising using camera image with synthetic noise
#     """
#
#     image = rescale_intensity(
#         camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
#     )
#
#     intensity = 5
#     numpy.random.seed(0)
#     noisy = numpy.random.poisson(image * intensity) / intensity
#     noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
#     noisy = numpy.expand_dims(
#         numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
#     )
#     it = ImageTranslatorCNN(training_architecture=False)
#
#     start = time.time()
#     it.train(noisy, noisy)  # , max_epochs=30
#     stop = time.time()
#     print(f"####### Training: elapsed time:  {stop-start} sec")
#
#     start = time.time()
#     denoised = it.translate(noisy)
#     stop = time.time()
#     print(f"####### Inference: elapsed time:  {stop-start} sec")
#
#     image = numpy.clip(image, 0, 1)
#     noisy = numpy.clip(noisy.squeeze(), 0, 1)
#     denoised = numpy.clip(denoised.squeeze(), 0, 1)
#
#     psnr_noisy = psnr(noisy, image)
#     ssim_noisy = ssim(noisy, image)
#     print("noisy", psnr_noisy, ssim_noisy)
#
#     psnr_denoised = psnr(denoised, image)
#     ssim_denoised = ssim(denoised, image)
#     print("denoised", psnr_denoised, ssim_denoised)
#
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#
#     # PSNR and SSIM may be lower than those of classic methods.
#     # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
#     # do not change the number below, but instead, fix the problem -- most likely a parameter.
#
#     assert psnr_denoised > 23 and ssim_denoised > 0.81
#
#
# @pytest.mark.heavy
# def test_it_cnn_supervised():
#     """
#         Test for self-supervised denoising using camera image with synthetic noise
#     """
#
#     image = rescale_intensity(
#         camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
#     )
#
#     intensity = 5
#     numpy.random.seed(0)
#     noisy = numpy.random.poisson(image * intensity) / intensity
#     noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
#     noisy = numpy.expand_dims(
#         numpy.expand_dims(noisy.astype(numpy.float32), axis=2), axis=0
#     )
#     it = ImageTranslatorCNN(training_architecture=False)
#
#     start = time.time()
#     it.train(noisy, noisy, max_epochs=10)
#     stop = time.time()
#     print(f"####### Training: elapsed time:  {stop-start} sec")
#
#     start = time.time()
#     denoised = it.translate(noisy)
#     stop = time.time()
#     print(f"####### Inference: elapsed time:  {stop-start} sec")
#
#     image = numpy.clip(image, 0, 1)
#     noisy = numpy.clip(noisy.squeeze(), 0, 1)
#     denoised = numpy.clip(denoised.squeeze(), 0, 1)
#
#     psnr_noisy = psnr(noisy, image)
#     ssim_noisy = ssim(noisy, image)
#     print("noisy", psnr_noisy, ssim_noisy)
#
#     psnr_denoised = psnr(denoised, image)
#     ssim_denoised = ssim(denoised, image)
#     print("denoised", psnr_denoised, ssim_denoised)
#
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#     assert psnr_denoised > psnr_noisy and ssim_denoised > ssim_noisy
#
#     # PSNR and SSIM may be lower than those of classic methods.
#     # if the line below fails, then something has be be wrong with CNN (e.g. it_cnn).
#     # do not change the number below, but instead, fix the problem -- most likely a parameter.
#
#     assert psnr_denoised > 23 and ssim_denoised > 0.81
