# flake8: noqa
import time
from datetime import date
from os.path import join


import numpy

from aydin.features.fast.fast_features import FastMultiscaleConvolutionalFeatures
from aydin.features.tiled.tiled_features import TiledFeatureGenerator
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import imwrite
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor


def demo():

    today = date.today()
    filename = f"result_{today.strftime('%m_%d_%Y_%H_%M_%S')}.tiff"

    # (3, 320, 865, 1014)
    image_path = examples_single.cognet_nanotube3.get_path()
    array, metadata = io.imread(image_path)
    print(array.shape)
    train = array[0:120]
    infer = array[0:120]

    # print(f"Number of distinct features in image: {len(numpy.unique(infer))}")

    print(f"train: {train.shape}, inference:{infer.shape} ")

    batch_dims = (False, False, False)

    generator = FastMultiscaleConvolutionalFeatures(  # kernel_widths=[3, 3, 1, 1, 1,  1,  1,  1,   1,   1],
        # kernel_scales=[0, 1, 2, 3, 7, 15, 31, 63, 127, 255],
        # kernel_shapes=['li'] * 2 + ['l1'] * 8,
        max_level=9,
        dtype=numpy.float16,
        exclude_scale_one=True,
        include_median_features=False,
    )
    generator = TiledFeatureGenerator(generator, max_tile_size=512)

    regressor = NNRegressor(max_epochs=200, patience=15)

    it = ImageTranslatorClassic(generator, regressor, normaliser_type='percentile')

    start = time.time()
    it.train(train, train, batch_dims=batch_dims)
    stop = time.time()
    print(f"Training: elapsed time:  {stop-start} ")

    output_file = join(get_temp_folder(), filename)

    print(f"Output file: {output_file}")

    # We write the stack to a temp file:
    with imwrite(output_file, infer.shape, infer.dtype) as denoised_tiff:

        # denoised = offcore_array(whole.shape, whole.dtype)

        start = time.time()
        denoised = it.translate(
            infer, translated_image=denoised_tiff, batch_dims=batch_dims, tile_size=512
        )
        stop = time.time()

        print(f"Writing to file: {output_file} ")
        denoised_tiff[...] = denoised[...]

        import napari

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(train, name='train')
            viewer.add_image(infer, name='infer')
            print(f"Inference: elapsed time:  {stop-start} ")
            viewer.add_image(denoised, name='denoised')


demo()
