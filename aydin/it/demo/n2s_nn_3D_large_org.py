import time
from os.path import join

import napari
import numpy

from aydin.features.fast.mcfoclf import FastMultiscaleConvolutionalFeatures
from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.io.folders import get_temp_folder
from aydin.io.io import imwrite
from aydin.it.it_classic import ImageTranslatorClassic
from aydin.regression.nn import NNRegressor


def demo():

    # (3, 320, 865, 1014)
    image_path = examples_single.gardner_org.get_path()
    array, metadata = io.imread(image_path)
    print(array.shape)
    array = array.squeeze()
    array = array[1]

    train = array  # full
    # train = array[50:250, 300:500, 400:600]

    infer = array  # Full: 320, 865, 1014
    # infer = array[0:160, 0:430, 0:512] # 1/8th

    # print(f"Number of distinct features in image: {len(numpy.unique(infer))}")

    print(f"train: {train.shape}, inference:{infer.shape} ")

    with napari.gui_qt():

        viewer = napari.Viewer()
        viewer.add_image(train, name='train')
        viewer.add_image(infer, name='infer')

        batch_dims = (False,) * len(array.shape)

        generator = FastMultiscaleConvolutionalFeatures(max_level=4, dtype=numpy.uint16)
        regressor = NNRegressor()

        it = ImageTranslatorClassic(
            generator,
            regressor,
            normaliser_type='percentile',
            balance_training_data=True,
        )

        start = time.time()
        it.train(
            train,
            train,
            batch_dims=batch_dims,
            batch_size=100e6,
            max_epochs=30,
            patience=8,
        )
        stop = time.time()
        print(f"Training: elapsed time:  {stop-start} ")

        output_file = join(get_temp_folder(), "result.tiff")

        print(f"Output file: {output_file}")

        # We write the stack to a temp file:
        with imwrite(output_file, infer.shape, infer.dtype) as denoised_tiff:

            # denoised = offcore_array(whole.shape, whole.dtype)

            start = time.time()
            denoised = it.translate(
                infer, translated_image=denoised_tiff, batch_dims=batch_dims
            )
            stop = time.time()

            print(f"Writing to file: {output_file} ")
            denoised_tiff[...] = denoised[...]

            print(f"Inference: elapsed time:  {stop-start} ")
            viewer.add_image(denoised, name='denoised')


demo()
