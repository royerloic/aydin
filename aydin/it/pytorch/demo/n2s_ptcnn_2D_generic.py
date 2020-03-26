# flake8: noqa
import os
import time

import numpy
from skimage.data import camera
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim

from aydin.analysis.image_metrics import mutual_information
from aydin.io.datasets import (
    normalise,
    add_noise,
    newyork,
    characters,
    lizard,
    scafoldings,
    pollen,
    andromeda,
    fibsem,
    dots,
)
from aydin.it.pytorch.it_ptcnn import PTCNNImageTranslator


def demo(image, name, do_add_noise=True):
    # image = image[0:1024, 0:1024]

    image = normalise(image.astype(numpy.float32))

    noisy = (
        add_noise(image, intensity=100, variance=0.1, sap=0.01)
        if do_add_noise
        else image
    )
    # noisy_image = image

    it = PTCNNImageTranslator(
        max_epochs=1000, patience=256, learning_rate=0.01, normaliser_type='identity'
    )

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"Training: elapsed time:  {stop - start} ")

    # in case of batching we have to do this:
    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"inference: elapsed time:  {stop - start} ")

    print(denoised.shape)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    denoised = numpy.clip(denoised, 0, 1)

    psnr_noisy = psnr(image, noisy)
    mi_noisy = mutual_information(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    mi_denoised = mutual_information(image, denoised)
    ssim_denoised = ssim(image, denoised)

    print("noisy       :", psnr_noisy, mi_noisy, ssim_noisy)
    print("denoised_inf:", psnr_denoised, mi_denoised, ssim_denoised)

    # visualise weights with napari:
    # it.visualise_weights()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(2.7 * 5, 5), dpi=300)
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
    os.makedirs("../../demo/demo_results", exist_ok=True)
    plt.savefig(f'demo_results/n2s_jidcnet_2D_{name}.png')

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(normalise(image), name='image')
        viewer.add_image(normalise(noisy), name='noisy')
        viewer.add_image(normalise(denoised), name='denoised')

    return ssim(denoised, image)


dots_image = dots()
demo(dots_image, "dots")
camera_image = camera()
demo(camera_image, "camera")
lizard_image = lizard()
demo(lizard_image, "lizard")
pollen_image = pollen()
demo(pollen_image, "pollen")
newyork_image = newyork()
demo(newyork_image, "newyork")
characters_image = characters()
demo(characters_image, "characters")
scafoldings_image = scafoldings()
demo(scafoldings_image, "scafoldings")
andromeda_image = andromeda()
demo(andromeda_image, "andromeda")
fibsem_image = fibsem()
demo(fibsem_image, "fibsem", False)
