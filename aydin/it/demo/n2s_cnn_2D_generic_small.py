import time

import numpy
import skimage
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise

from aydin.io.datasets import normalise, add_noise, lizard, pollen, newyork
from aydin.it.it_cnn import ImageTranslatorCNN


def demo(image, max_epochs=4, image_width=200):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(image)
    H0, W0 = (numpy.array(image.shape) - image_width) // 2
    image = image[H0 : H0 + image_width, W0 : W0 + image_width]
    noisy = add_noise(image)
    noisy = numpy.expand_dims(numpy.expand_dims(noisy, axis=2), axis=0)

    # Classical denoisers:
    # median1 = skimage.filters.median(noisy, disk(1))
    # median2 = skimage.filters.median(noisy, disk(2))
    # median5 = skimage.filters.median(noisy, disk(5))
    # nlm = denoise_nl_means(noisy, patch_size=11, sigma=estimate_sigma(noisy))

    # CNN based Image translation:
    # input_dim only includes H, W, C; number of images is not included
    it = ImageTranslatorCNN(
        training_architecture='shiftconv',
        num_layer=3,
        batch_norm=None,  # 'instance',
        max_epochs=max_epochs,
        verbose=1,
    )

    start = time.time()
    # total_num_patches decides how many tiling batches to train.
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time: {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy, tile_size=image_width)
    denoised_inf = denoised_inf.reshape(image.shape)
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
        # viewer.add_image(n(nlm), name='nlm')
        # viewer.add_image(n(median1), name='median1')
        # viewer.add_image(n(median2), name='median2')
        # viewer.add_image(n(median5), name='median5')
        viewer.add_image(normalise(denoised_inf), name='denoised_inf')


camera_image = camera()
demo(camera_image)
