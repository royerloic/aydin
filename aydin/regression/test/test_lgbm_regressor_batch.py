import math
import pytest

import napari
import numpy
from skimage.data import camera
from skimage.exposure import rescale_intensity
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from skimage.util import random_noise

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.regression.gbm import GBMRegressor


def lgbm_regressor(batch, num_batches=10, num_used_batches=math.inf, display=True):
    image = camera().astype(numpy.float32)
    image = rescale_intensity(image, in_range='image', out_range=(0, 1))

    intensity = 5
    numpy.random.seed(0)
    noisy = numpy.random.poisson(image * intensity) / intensity
    noisy = random_noise(noisy, mode='gaussian', var=0.01, seed=0)
    noisy = noisy.astype(numpy.float32)

    scales = [1, 3, 7, 15]
    widths = [3, 3, 3, 3]

    generator = FastMultiscaleConvolutionalFeatures(
        kernel_widths=widths, kernel_scales=scales, kernel_shapes=['l1'] * len(scales)
    )

    features = generator.compute(noisy)

    x = features.reshape(-1, features.shape[-1])
    y = noisy.reshape(-1)

    # shuffling:
    num_datapoints = len(y)
    perm = numpy.random.permutation(num_datapoints)
    xt = x[perm, :][num_datapoints // 10 :]
    yt = y[perm][num_datapoints // 10 :]
    xv = x[perm, :][0 : num_datapoints // 10]
    yv = y[perm][0 : num_datapoints // 10]

    params = {
        'learning_rate': 0.01,
        'num_leaves': 127,
        'max_bin': 512,
        'n_estimators': 512,
        'patience': 20,
        'verbosity': 0,
    }

    regressor = GBMRegressor(**params)

    if batch:

        x_batches = numpy.array_split(xt, num_batches)
        y_batches = numpy.array_split(yt, num_batches)

        batch_counter = 0

        for x_batch, y_batch in zip(x_batches, y_batches):

            regressor.fit(x_batch, y_batch, xv, yv, is_batch=True)

            batch_counter = batch_counter + 1
            if batch_counter > num_used_batches:
                break

    else:
        regressor.fit(x, y, xv, yv)

    yp = regressor.predict(x)

    denoised = yp.reshape(image.shape)
    denoised = rescale_intensity(denoised, in_range='image', out_range=(0, 1))

    ssim_value = ssim(denoised, image)
    psnr_value = psnr(denoised, image)

    print("denoised", psnr, ssim)

    if display:
        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(
                rescale_intensity(image, in_range='image', out_range=(0, 1)),
                name='image',
            )
            viewer.add_image(denoised, name='denoised')

    return ssim_value


@pytest.mark.heavy
def test_demo_lgbm_regressor():
    ssim_no_batch = lgbm_regressor(batch=False, display=False)
    ssim_one_batch = lgbm_regressor(batch=True, num_used_batches=1, display=False)
    ssim_batch = lgbm_regressor(batch=True, display=False)

    print(
        f"ssim_no_batch={ssim_no_batch}, ssim_one_batch={ssim_one_batch}, ssim_batch={ssim_batch} "
    )

    assert ssim_no_batch > ssim_batch
    assert ssim_batch > ssim_one_batch
    assert ssim_one_batch > 0.79
    assert ssim_batch > 0.81
