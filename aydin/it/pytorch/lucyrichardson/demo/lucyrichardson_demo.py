# flake8: noqa

import numpy
import torch
from numpy.linalg import norm
from scipy import ndimage, rand
from scipy.signal import convolve2d
from skimage import restoration
from skimage.measure import compare_ssim as ssim

from aydin.io.datasets import normalise, add_noise, characters
from aydin.it.it_skipnet import SkipNetImageTranslator
from aydin.it.pytorch.models.lucyrichardson import LucyRichardson
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def demo(image):

    image = 1 - normalise(image.astype(numpy.float32))

    psf = SimpleMicroscopePSF()
    psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
    print(psf_xyz_array.shape)
    kernel_psf = psf_xyz_array[0]

    blurred_image = convolve2d(image, kernel_psf, 'same')

    noisy_and_blurred_image = add_noise(blurred_image)

    module_deconvolved_image = (
        LucyRichardson(kernel_psf=kernel_psf, iterations=10)(
            torch.from_numpy(
                noisy_and_blurred_image[numpy.newaxis, numpy.newaxis, ...]
            ).float()
        )
        .detach()
        .cpu()
        .numpy()
    )

    it = SkipNetImageTranslator(max_epochs=512, patience=128)

    it.train(noisy_and_blurred_image)
    denoised = it.translate(noisy_and_blurred_image)

    lr_max_iterations = 60

    input_image = denoised
    mask = numpy.random.rand(*image.shape) < 0.1
    masked_input_image = (
        input_image * ~mask
        + ndimage.median_filter(
            input_image, footprint=numpy.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        )
        * mask
    )
    deconvolved_image = numpy.stack(
        [
            richardson_lucy_pytorch(masked_input_image, kernel_psf, iterations=i)
            for i in range(0, lr_max_iterations, 1)
        ]
    )

    reconvolved_image = numpy.stack(
        [convolve2d(i, kernel_psf, 'same') for i in deconvolved_image]
    )

    self_supervised_loss = [
        norm(i * mask - j * mask, 2) for i, j in zip(reconvolved_image, input_image)
    ]

    print(self_supervised_loss)
    print(self_supervised_loss.index(min(self_supervised_loss)))

    supervised_loss = [1 - ssim(i, image) for i in deconvolved_image]
    print(supervised_loss)
    print(supervised_loss.index(min(supervised_loss)))

    # pt_denoised_deconvolved_image = numpy.stack([richardson_lucy_pytorch(denoised, kernel_psf, iterations=i)for i in range(0,40,1)])

    # it.train(si_deconvolved_image)
    # si_deconvolved_denoised_image = it.translate(si_deconvolved_image)

    import napari

    with napari.gui_qt():
        viewer = napari.Viewer()
        # viewer.add_image(module_deconvolved_image, name='module_deconvolved_image')
        # viewer.add_image(si_deconvolved_denoised_image, name='si_deconvolved_denoised_image')
        # viewer.add_image(pt_denoised_deconvolved_image, name='pt_denoised_deconvolved_image')
        # viewer.add_image(denoised, name='denoised')
        viewer.add_image(reconvolved_image, name='reconvolved_image')
        viewer.add_image(deconvolved_image, name='deconvolved_image')
        viewer.add_image(noisy_and_blurred_image, name='noisy_and_blurred_image')
        viewer.add_image(image, name='image')


def richardson_lucy_pytorch(image, psf, iterations=50, clip=True):

    import torch.nn.functional as F
    import torch

    use_cuda = True
    device_index = 0
    device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
    # print(f"Using device: {device}")

    image = image.astype(numpy.float)
    psf = psf.astype(numpy.float)
    im_deconv = numpy.full(image.shape, 0.5)
    psf_mirror = psf[::-1, ::-1].copy()
    psf_size = psf_mirror.shape[0]

    image = (
        torch.from_numpy(image[numpy.newaxis, numpy.newaxis, ...]).float().to(device)
    )
    psf = torch.from_numpy(psf[numpy.newaxis, numpy.newaxis, ...]).float().to(device)
    psf_mirror = (
        torch.from_numpy(psf_mirror[numpy.newaxis, numpy.newaxis, ...])
        .float()
        .to(device)
    )
    im_deconv = (
        torch.from_numpy(im_deconv[numpy.newaxis, numpy.newaxis, ...])
        .float()
        .to(device)
    )

    for _ in range(iterations):
        convolved = F.conv2d(im_deconv, psf, padding=(psf_size - 1) // 2)
        relative_blur = image / convolved
        im_deconv *= F.conv2d(relative_blur, psf_mirror, padding=(psf_size - 1) // 2)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv.detach().cpu().numpy().squeeze()


image = characters()
demo(image)
