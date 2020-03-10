# flake8: noqa
import time

import numpy
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.io.datasets import normalise, add_noise, lizard, pollen, newyork
from aydin.it.it_cnn import ImageTranslatorCNN


def demo(image, max_epochs=10):

    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    # Classical denoisers:
    # median1 = skimage.filters.median(noisy, disk(1))
    # median2 = skimage.filters.median(noisy, disk(2))
    # median5 = skimage.filters.median(noisy, disk(5))
    # nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    # CNN based Image translation:
    # input_dim only includes H, W, C; number of images is not included
    it = ImageTranslatorCNN(
        training_architecture='checkerbox',
        num_layer=5,
        batch_norm='instance',  # None,  #
        activation='ReLU',
        tile_size=128,
        mask_shape=(3, 3),
        # total_num_patches=10,
        max_epochs=max_epochs,
        verbose=1,
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy, tile_size=512)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy.reshape(image.shape), 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)
    print("noisy       :", psnr(image, noisy), ssim(noisy, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised_inf), name='denoised_inf')


camera_image = camera()
demo(camera_image)
lizard_image = lizard()
demo(lizard_image)
pollen_image = pollen()
demo(pollen_image)
newyork_image = newyork()
demo(newyork_image)
