# flake8: noqa
import time

import numpy
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.analysis.image_metrics import mutual_information, spectral_mutual_information
from aydin.io.datasets import normalise, add_noise, characters, add_blur_2d
from aydin.it.it_lr_deconv import LucyRichardsonDeconvolution


def demo(image):

    image = normalise(image.astype(numpy.float32))
    blurred_image, psf_kernel = add_blur_2d(image)
    noisy_and_blurred_image = add_noise(
        blurred_image, intensity=100, variance=0.1, sap=0.01
    )

    lr = LucyRichardsonDeconvolution(psf_kernel=psf_kernel, max_num_iterations=20)

    start = time.time()
    lr.train(noisy_and_blurred_image, noisy_and_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    start = time.time()
    lr_deconvolved_image = lr.translate(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    print("Below in order: PSNR, norm spectral mutual info, norm mutual info, SSIM: ")

    def printscore(header, val1, val2, val3, val4):
        print(f"{header}: \t {val1:.4f} \t {val2:.4f} \t {val3:.4f} \t {val4:.4f}")

    printscore(
        "n&b image",
        psnr(image, noisy_and_blurred_image),
        spectral_mutual_information(image, noisy_and_blurred_image),
        mutual_information(image, noisy_and_blurred_image),
        ssim(image, noisy_and_blurred_image),
    )
    # printscore(
    #     "den image",
    #     psnr(image, denoised_image),
    #     spectral_mutual_information(image, denoised_image),
    #     mutual_information(image, denoised_image),
    #     ssim(image, denoised_image),
    # )
    printscore(
        "lr      ",
        psnr(image, lr_deconvolved_image),
        spectral_mutual_information(image, lr_deconvolved_image),
        mutual_information(image, lr_deconvolved_image),
        ssim(image, lr_deconvolved_image),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(blurred_image, name='blurred')
        viewer.add_image(noisy_and_blurred_image, name='noisy')
        viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')


image = characters()

demo(image)
