import time
import numpy
import pytest
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from aydin.io.datasets import normalise, add_noise
from aydin.it.it_cnn import ImageTranslatorCNN


def test_it_cnn_shiftconv_light():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 2
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
    max_epochs = 2
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
    max_epochs = 5
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    it = ImageTranslatorCNN(
        training_architecture='random',
        num_layer=3,
        batch_norm='instance',
        max_epochs=max_epochs,
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


def test_it_cnn_checkran_light():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 2
    image_width = 200
    image = normalise(camera())
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    it = ImageTranslatorCNN(
        training_architecture='checkran',
        num_layer=2,
        mask_shape=(3, 3),
        batch_norm='instance',
        max_epochs=max_epochs,
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


@pytest.mark.heavy
def test_it_cnn_random_patching():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    start = time.time()
    max_epochs = 8
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
        patch_size=64,
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
