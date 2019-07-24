# Example of using VST
import time

import napari
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from pitl.features.classic.mcfocl import MultiscaleConvolutionalFeatures
from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.normaliser.vst.vst import vst_classical, vst_transform, vst_adaptive
from pitl.regression.gbm import GBMRegressor

image = camera().astype(numpy.float32)  # import image
image = rescale_intensity(image, in_range='image', out_range=(0, 1))

intensity = (
    5.0
)  # strength of noise, which can also be measured from the ratio between variance of noise and mean

noisy = numpy.random.poisson(image * intensity) / intensity  # Poisson Noise
noisy = noisy.astype(numpy.float32)
noisy2 = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
noisy2 = noisy2.astype(numpy.float32)

# Freeman Tukey Transform
x, y = vst_classical(image, intensity)

# apply VST
noisy_t = vst_transform(
    noisy2, x, y / y[-1]
)  # y normalized between [0,1] so that after transform, image is in [0,1]

# train regressor
scales = [1, 3, 7, 15, 31, 63, 127, 255]
widths = [3, 3, 3, 3, 3, 3, 3, 3]

generator = FastMultiscaleConvolutionalFeatures(
    kernel_widths=widths,
    kernel_scales=scales,
    kernel_shapes=['l1'] * len(scales),
    exclude_center=True,
)

regressor = GBMRegressor(
    learning_rate=0.01,
    num_leaves=127,
    max_bin=512,
    n_estimators=2048,
    early_stopping_rounds=20,
)


it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

denoised = it.train(noisy2, noisy2)  # regress no VST data
denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

denoised_t = it.train(noisy_t, noisy_t)  # regress VST data


# inverse transform
denoised_t = vst_transform(denoised_t, y / y[-1], x)
denoised_t = rescale_intensity(denoised_t, in_range='image', out_range=(0, 1))

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(noisy, name='noisy')
    viewer.add_image(noisy_t, name='noisy VST')
    viewer.add_image(noisy2, name='noisy2')
    viewer.add_image(denoised, name='no VST')
    viewer.add_image(denoised_t, name='VST')

# the adaptive transform can be calculated from the noisy and denoised images
x, y = vst_adaptive(noisy, denoised)
# one can then go back to the apply VST step and re do the training process
