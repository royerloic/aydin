import numpy
from scipy.signal import convolve2d
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.restoration import richardson_lucy

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.io.datasets import characters, add_noise
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.util.psf.simple_microscope_psf import SimpleMicroscopePSF


def n(image):
    return rescale_intensity(
        image.astype(numpy.float32), in_range='image', out_range=(0, 1)
    )


# Prepare image:
image = 1 - characters()
image = image.astype(numpy.float32)
image = n(image)

# Prepare PSF:
psf = SimpleMicroscopePSF()
psf_xyz_array = psf.generate_xyz_psf(dxy=0.406, dz=0.406, xy_size=17, z_size=17)
print(psf_xyz_array.shape)
kernel_psf = psf_xyz_array[0]

# degrade image with blurr and noise;
blurred_image = convolve2d(image, kernel_psf, 'same')
noisy_and_blurred_image = add_noise(blurred_image)

# try classic LR:
lr_deconvolved_image = richardson_lucy(
    noisy_and_blurred_image, kernel_psf, iterations=30
)

# Try denoising and then lr deconvolution:
it = ImageTranslatorClassic(
    feature_generator=FastMultiscaleConvolutionalFeatures(),
    regressor=GBMRegressor(n_estimators=3000),
    normaliser_type='identity',
)
it.train(noisy_and_blurred_image)
denoised_image = it.translate(noisy_and_blurred_image)
denoised_deconvolved_image = richardson_lucy(denoised_image, kernel_psf, iterations=30)

# lr deconvolution followed by denoising:
it.train(lr_deconvolved_image)
deconvolved_denoised_image = it.translate(lr_deconvolved_image)


# Train to translate input to deconvolved, but with blind spot:

# Generating features:
generator = FastMultiscaleConvolutionalFeatures(exclude_scale_one=False)
features = generator.compute(noisy_and_blurred_image, exclude_center_value=False)
x = features.reshape(-1, features.shape[-1])
y = denoised_deconvolved_image.reshape(-1)

# Training:
regressor = GBMRegressor(n_estimators=3000)
regressor.fit(x, y, x_valid=x, y_valid=y)

# Computing features again but without blind-spot:
features = generator.compute(noisy_and_blurred_image, exclude_center_value=False)
x = features.reshape(-1, features.shape[-1])

# predict denoised and deconvolved image:
yp = regressor.predict(x)
restored_image = yp.reshape(image.shape)

# Clipping for comparison:
lr_deconvolved_image = numpy.clip(lr_deconvolved_image, 0, 1)
denoised_image = numpy.clip(denoised_image, 0, 1)
deconvolved_denoised_image = numpy.clip(deconvolved_denoised_image, 0, 1)
denoised_deconvolved_image = numpy.clip(denoised_deconvolved_image, 0, 1)
restored_image = numpy.clip(restored_image, 0, 1)

# Compare results:
print(
    "lr_deconvolved_image",
    psnr(lr_deconvolved_image, image),
    ssim(lr_deconvolved_image, image),
)
print("denoised_image", psnr(denoised_image, image), ssim(denoised_image, image))
print(
    "deconvolved_denoised_image",
    psnr(deconvolved_denoised_image, image),
    ssim(deconvolved_denoised_image, image),
)
print(
    "denoised_deconvolved_image",
    psnr(denoised_deconvolved_image, image),
    ssim(denoised_deconvolved_image, image),
)
print("restored_image", psnr(restored_image, image), ssim(restored_image, image))

import napari

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(blurred_image, name='blurred_image')
    viewer.add_image(noisy_and_blurred_image, name='noisy_and_blurred_image')
    viewer.add_image(lr_deconvolved_image, name='lr_deconvolved_image')
    viewer.add_image(denoised_image, name='denoised_image')
    viewer.add_image(deconvolved_denoised_image, name='deconvolved_denoised_image')
    viewer.add_image(denoised_deconvolved_image, name='denoised_deconvolved_image')
    viewer.add_image(restored_image, name='restored_image')
