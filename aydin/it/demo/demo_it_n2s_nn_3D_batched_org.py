import time

import napari
import numpy
from skimage.exposure import rescale_intensity

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.gbm import GBMRegressor
from aydin.regression.nn.nn import NNRegressor


def demo(image):

    with napari.gui_qt():

        batch_dims = (True, False, False, False)

        generator = FastMultiscaleConvolutionalFeatures(max_features=30)

        regressor = NNRegressor()

        it = ImageTranslatorClassic(generator, regressor, normaliser='identity')

        start = time.time()
        it.train(image, image, batch_dims=batch_dims)
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        start = time.time()
        denoised = it.translate(image, batch_dims=batch_dims)
        stop = time.time()
        print(f"inference train: elapsed time:  {stop-start} ")

        viewer = napari.Viewer()
        viewer.add_image(image, name='image')
        viewer.add_image(denoised, name='denoised')

        print(image.shape)
        print(denoised.shape)


# (3, 320, 865, 1014)
image_path = examples_single.gardner_org.get_path()
array, metadata = io.imread(image_path)
print(array.shape)
array = array.squeeze()
array = array[:, 0:60, 270:500, 400:600]
# array = array[:, 0:60, 170:600, 300:700]
print(array.shape)
array = array.astype(numpy.float32)
array = rescale_intensity(array, in_range='image', out_range=(0, 1))
demo(array)
