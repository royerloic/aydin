import time

import napari
import numpy
from skimage.exposure import rescale_intensity

from pitl.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from pitl.io import io
from pitl.io.datasets import examples_single
from pitl.it.it_classic import ImageTranslatorClassic
from pitl.offcore.offcore import offcore_array
from pitl.regression.gbm import GBMRegressor
from pitl.regression.nn.nn import NNRegressor


def demo():

    # (3, 320, 865, 1014)
    image_path = examples_single.gardner_org.get_path()
    array, metadata = io.imread(image_path)
    print(array.shape)
    array = array.squeeze()
    array = array[1]

    train = array  # full
    # train = array[100:200, 200:600, 300:700]
    # train = array[0:10, 170:200, 300:310]  # very_mini_tiny

    whole = array  # Full: 320, 865, 1014
    # whole = array[0:160, 0:430, 0:512] # 1/8th

    print(f"Number of distinct features in image: {len(numpy.unique(whole))}")

    print(f"train: {train.shape}, inference:{whole.shape} ")

    with napari.gui_qt():

        viewer = napari.Viewer()
        viewer.add_image(train, name='train')
        viewer.add_image(whole, name='image')

        level = 4
        scales = [1, 3, 7, 15, 31][:level]
        widths = [3, 3, 3, 3, 3][:level]

        batch_dims = (False, False, False)

        generator = FastMultiscaleConvolutionalFeatures(
            kernel_widths=widths, kernel_scales=scales
        )

        regressor = NNRegressor()

        it = ImageTranslatorClassic(generator, regressor, normaliser='percentile')

        start = time.time()
        it.train(train, train, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        denoised = offcore_array(whole.shape, whole.dtype)

        start = time.time()
        denoised = it.translate(whole, translated_image=denoised, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")
        viewer.add_image(denoised, name='denoised')


demo()
