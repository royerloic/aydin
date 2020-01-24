import time

# import napari
import numpy
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor

"""
    Demo for self-supervised denoising using camera image with synthetic noise
"""


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


def demo():

    image = camera().astype(np.float32)
    image = n(image)
    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    start_time = time.time()

    generator = FastMultiscaleConvolutionalFeatures()
    regressor = NNRegressor()

    it = ImageTranslatorClassic(
        feature_generator=generator, regressor=regressor, normaliser_type='identity'
    )

    it.train(noisy, noisy, max_epochs=30, patience=10)

    elapsedtime = time.time() - start_time
    print(f"time elapsed: {elapsedtime} s")

    start = time.time()
    denoised_inf = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised_inf, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(noisy, image)
    psnr_denoised = psnr(image, denoised_inf)
    ssim_denoised = ssim(denoised_inf, image)
    print("noisy       :", psnr_noisy, ssim_noisy)
    print("denoised_inf:", psnr_denoised, ssim_denoised)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(n(noisy), cmap='gray')
    plt.axis('off')
    plt.title(f'Noisy \nPSNR: {psnr_noisy:.3f}, SSIM: {ssim_noisy:.3f}')
    plt.subplot(1, 3, 2)
    plt.imshow(n(denoised_inf), cmap='gray')
    plt.axis('off')
    plt.title(f'Denoised \nPSNR: {psnr_denoised:.3f}, SSIM: {ssim_denoised:.3f}')
    plt.subplot(1, 3, 3)
    plt.imshow(n(image), cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.1)
    plt.savefig(f'n2s_nn_2D.png')

    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(n(image), name='image')
    #     viewer.add_image(n(noisy), name='noisy')
    #     viewer.add_image(n(denoised), name='denoised')


demo()
