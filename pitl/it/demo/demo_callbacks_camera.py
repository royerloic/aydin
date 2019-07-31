import time

import napari
import numpy as np
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from pitl.cli.progress_bar import ProgressBar
from pitl.services.n2s import N2SService


def demo(image):
    """
        Demo for self-supervised denoising using camera image with synthetic noise
    """
    image = image.astype(np.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    np.random.seed(0)
    noisy = np.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(np.float32)

    with napari.gui_qt():
        viewer = napari.Viewer()

        scales = [1, 3, 7, 15, 31, 63, 127, 255]
        widths = [3, 3, 3, 3, 3, 3, 3, 3]

        size = 128
        monitoring_image = noisy[
            256 - size // 2 : 256 + size // 2, 256 - size // 2 : 256 + size // 2
        ]

        def emit_func(arg):
            print(arg)
            image, eval_metric, iter = arg
            print(f"Iteration: {iter} metric: {eval_metric}")
            # print(f"images: {str(images)}")
            print("image: ", image[0])
            if image[0] is not None:
                viewer.add_image(
                    rescale_intensity(image[0], in_range='image', out_range=(0, 1)),
                    name='noisy',
                )

        n2s = N2SService(scales, widths, emit_func)
        n2s.monitoring_variables = None, 0, -1

        start = time.time()
        pbar = ProgressBar(total=100)
        denoised = n2s.run(noisy, pbar, monitoring_images=[monitoring_image])
        pbar.close()
        # denoised = it.train(
        #     noisy,
        #     noisy,
        #     callbacks=[callback],
        #     # monitoring_images=[monitoring_image]
        # )

        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

        print("noisy", psnr(image, noisy), ssim(noisy, image))
        print("denoised", psnr(image, denoised), ssim(denoised, image))
        # print("denoised_predict", psnr(denoised_predict, image), ssim(denoised_predict, image))

        viewer.add_image(
            rescale_intensity(image, in_range='image', out_range=(0, 1)), name='image'
        )
        viewer.add_image(
            rescale_intensity(noisy, in_range='image', out_range=(0, 1)), name='noisy'
        )
        viewer.add_image(
            rescale_intensity(denoised, in_range='image', out_range=(0, 1)),
            name='denoised',
        )
        # viewer.add_image(rescale_intensity(denoised_predict, in_range='image', out_range=(0, 1)), name='denoised_predict%d' % param)


demo(camera())
