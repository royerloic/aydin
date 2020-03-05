# flake8: noqa
import os
import time


import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import normalise, add_noise, pollen, newyork, lizard
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.rf import RandomForrestRegressor


def demo(image, name):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """

    image = normalise(image.astype(numpy.float32))
    noisy = add_noise(image)

    generator = FastMultiscaleConvolutionalFeatures()

    regressor = RandomForrestRegressor()

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    denoised = it.translate(noisy)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("noisy   :", psnr_noisy, ssim_noisy)
    print("denoised:", psnr_denoised, ssim_denoised)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(normalise(noisy), cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy \nPSNR: {psnr_noisy:.3f}, SSIM: {ssim_noisy:.3f}')
    plt.subplot(1, 3, 2)
    plt.imshow(normalise(denoised), cmap='gray')
    plt.axis('off')
    plt.title(f'Denoised \nPSNR: {psnr_denoised:.3f}, SSIM: {ssim_denoised:.3f}')
    plt.subplot(1, 3, 3)
    plt.imshow(normalise(image), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1)
    os.makedirs("demo_results", exist_ok=True)
    plt.savefig(f'demo_results/n2s_rf_2D_{name}.png')

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised), name='denoised')


camera_image = camera()
demo(camera_image, "camera")
lizard_image = lizard()
demo(lizard_image, "lizard")
pollen_image = pollen()
demo(pollen_image, "pollen")
newyork_image = newyork()
demo(newyork_image, "newyork")
