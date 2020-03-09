# flake8: noqa
import time


import numpy as np
from skimage.exposure import rescale_intensity

from aydin.io import io
from aydin.io.datasets import examples_single
from aydin.it.it_cnn import ImageTranslatorCNN

image_path = '../test_data/labeled_mararia/raw/p1.tif'
image, metadata = io.imread(image_path)
# image = image[0:10, 15:35, 130:167, 130:177]
# image = image[:, :, 100:300, 100:300]
image = np.expand_dims(np.expand_dims(image.astype(np.float32), axis=-1), axis=0)
image = rescale_intensity(image, in_range='image', out_range=(0, 1))
max_epochs = 10
# def demo(image, max_epochs=10):

batch_dims = (True, False, False, False, False)

it = ImageTranslatorCNN(
    training_architecture='random',
    num_layer=2,
    batch_norm='instance',  # None,  #
    activation='ReLU',
    mask_shape=(5, 5),
    tile_size=12,
    # total_num_patches=10,
    max_epochs=max_epochs,
    verbose=1,
)

start = time.time()
it.train(image, image, batch_dims=batch_dims)
stop = time.time()
print(f"Training: elapsed time:  {stop-start} ")

start = time.time()
denoised = it.translate(
    image,
    batch_dims=batch_dims,
    tile_size=256,  # image.shape[1:-1],  # [12, 12, 12],  # min(image.shape[1:-1])
)
stop = time.time()
print(f"inference: elapsed time:  {stop-start} ")
import napari

with napari.gui_qt():
    viewer = napari.Viewer()
    viewer.add_image(image, name='image')
    viewer.add_image(
        rescale_intensity(denoised, in_range='image', out_range=(0, 1)), name='denoised'
    )


# demo(image)
