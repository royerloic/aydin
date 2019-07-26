import time

import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.util import random_noise

from pitl.features.classic.mcfocl import MultiscaleConvolutionalFeatures
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.regression.gbm import GBMRegressor


def test_it_classic():
    """
        Test for self-supervised denoising using camera image with synthetic noise
    """

    image = rescale_intensity(
        camera().astype(numpy.float32), in_range='image', out_range=(0, 1)
    )

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    generator = MultiscaleConvolutionalFeatures(exclude_center=True)

    regressor = GBMRegressor(
        learning_rate=0.01,
        num_leaves=127,
        max_bin=512,
        n_estimators=2048,
        early_stopping_rounds=20,
    )

    it = ImageTranslatorClassic(feature_generator=generator, regressor=regressor)

    start = time.time()
    it.train(noisy, noisy)
    stop = time.time()
    print(f"####### Training: elapsed time:  {stop-start} sec")

    start = time.time()
    denoised = it.translate(noisy)
    stop = time.time()
    print(f"####### Inference: elapsed time:  {stop-start} sec")

    # TODO: implement callback test here!

    pass
