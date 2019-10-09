import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor


def demo_aydin_2D():
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = camera().astype(np.float32)  # [:,50:450]
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    scales = [1, 3, 5, 11, 21, 23, 47, 95]
    widths = [3, 3, 3, 3, 3, 3, 3, 3]

    for param in range(7, len(scales), 1):
        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths[0:param],
            kernel_scales=scales[0:param],
            kernel_shapes=['l1'] * len(scales[0:param]),
        )

        regressor = GBMRegressor(
            learning_rate=0.01, num_leaves=256, n_estimators=1024, patience=20
        )

        it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

        denoised = it.train(noisy, noisy)

        print("noisy", psnr(noisy, image), ssim(noisy, image))
        print("denoised", psnr(denoised, image), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)
