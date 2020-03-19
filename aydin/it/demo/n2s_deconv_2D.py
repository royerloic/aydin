# flake8: noqa
import time

import numpy
from scipy.signal import convolve2d
from skimage import restoration
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.io.datasets import normalise, add_noise, characters, scafoldings
from aydin.it.it_deconv import DeconvolvingImageTranslator
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def demo(image):

    image = normalise(image.astype(numpy.float32))

    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    print(psf_xyz_array.shape)
    psf_kernel = psf_xyz_array[0]

    blurred_image = convolve2d(image, psf_kernel, 'same')

    noisy_and_blurred_image = add_noise(blurred_image)

    lr_iterations = 30

    lr_deconvolved_image = restoration.richardson_lucy(
        noisy_and_blurred_image, psf_kernel, iterations=lr_iterations
    )
    lr_deconvolved_image = numpy.clip(lr_deconvolved_image, 0, 1)

    it = DeconvolvingImageTranslator(
        max_epochs=1000,
        patience=500,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=convolve2d(
            psf_kernel, numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) / 5, 'same'
        ),
    )

    start = time.time()
    it.train(noisy_and_blurred_image, noisy_and_blurred_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    aydin_denoised_and_deconvolved_image = it.translate(noisy_and_blurred_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    denoised_image = it.denoise(noisy_and_blurred_image)
    aydin_denoised_lr_deconvolved = restoration.richardson_lucy(
        denoised_image, psf_kernel, iterations=lr_iterations
    )

    image = numpy.clip(image, 0, 1)
    noisy_and_blurred_image = numpy.clip(noisy_and_blurred_image, 0, 1)
    aydin_denoised_and_deconvolved_image = numpy.clip(
        aydin_denoised_and_deconvolved_image, 0, 1
    )
    print(
        "noisy       :",
        psnr(image, noisy_and_blurred_image),
        ssim(noisy_and_blurred_image, image),
    )
    print(
        "aydin_denoised_and_deconvolved_image:",
        psnr(image, aydin_denoised_and_deconvolved_image),
        ssim(aydin_denoised_and_deconvolved_image, image),
    )
    print(
        "aydin_denoised_lr_deconvolved:",
        psnr(image, aydin_denoised_lr_deconvolved),
        ssim(aydin_denoised_lr_deconvolved, image),
    )
    print(
        "lr_deconvolved_image:",
        psnr(image, lr_deconvolved_image),
        ssim(lr_deconvolved_image, image),
    )

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(blurred_image), name='blurred')
        viewer.add_image(normalise(noisy_and_blurred_image), name='noisy')
        viewer.add_image(normalise(denoised_image), name='denoised_image')
        viewer.add_image(
            normalise(aydin_denoised_lr_deconvolved),
            name='aydin_denoised_lr_deconvolved',
        )
        viewer.add_image(normalise(lr_deconvolved_image), name='lr_deconvolved_image')
        viewer.add_image(
            normalise(aydin_denoised_and_deconvolved_image),
            name='aydin_denoised_and_deconvolved_image',
        )

    return ssim(aydin_denoised_and_deconvolved_image, image)


image = 1 - characters()
demo(image)
