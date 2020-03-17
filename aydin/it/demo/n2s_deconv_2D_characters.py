# flake8: noqa
import time

import numpy
from scipy.signal import convolve2d
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.io.datasets import normalise, add_noise, characters
from aydin.it.it_deconv import DeconvolvingImageTranslator
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def demo(image):

    # image = image[0:1024, 0:1024]

    image = 1 - normalise(image.astype(numpy.float32))

    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    print(psf_xyz_array.shape)
    psf_kernel = psf_xyz_array[0]

    blurred_image = convolve2d(image, psf_kernel, 'same')

    noisy_image = add_noise(blurred_image)
    # noisy_image = image

    # import napari
    # with napari.gui_qt():
    #     viewer = napari.Viewer()
    #     viewer.add_image(psf_kernel, name='psf_kernel')
    #     viewer.add_image(blurred_image, name='blurred_image')
    #     viewer.add_image(image, name='image')

    it = DeconvolvingImageTranslator(
        max_epochs=1000,
        learning_rate=0.01,
        normaliser_type='identity',
        psf_kernel=psf_kernel,
    )

    start = time.time()
    it.train(noisy_image, noisy_image)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised_inf = it.translate(noisy_image)
    stop = time.time()
    print(f"inference: elapsed time:  {stop-start} ")

    print(denoised_inf.shape)

    image = numpy.clip(image, 0, 1)
    noisy_image = numpy.clip(noisy_image, 0, 1)
    denoised_inf = numpy.clip(denoised_inf, 0, 1)
    print("noisy       :", psnr(image, noisy_image), ssim(noisy_image, image))
    print("denoised_inf:", psnr(image, denoised_inf), ssim(denoised_inf, image))

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy_image), name='noisy')
        viewer.add_image(normalise(blurred_image), name='blurred')
        viewer.add_image(normalise(denoised_inf), name='denoised')

    return ssim(denoised_inf, image)


image = characters()
demo(image)
# image = examples_single.scheffer_fibsem.get_array()[0:1024, 0:1024]
# ssims = []
# for i in range(6):
#     v = demo(image)
#     ssims.append(v)
# print(ssims)
# print(median(ssims))
# print(median(abs(x-median(ssims)) for x in ssims))
