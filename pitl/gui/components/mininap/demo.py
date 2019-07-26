import sys

import numpy as np
from vispy import app

from pitl.gui.components.mininap.gui.image_widget import ImageWidget
from pitl.gui.components.mininap.gui.image_window import ImageWindow
from pitl.gui.components.mininap.image.image_type_enum import ImageType
from pitl.gui.components.mininap.image.napari_image import NImage
from pitl.gui.components.mininap.napari_application import NapariApplication
from pitl.gui.components.mininap.utils.example_data_utils import load_bluemarble_image

app.use_app('pyqt5')


def open_2Drgb():
    # opening a 2D RGB image:
    bma = load_bluemarble_image(large=False)
    image = NImage(bma, 'BlueMarble', type=ImageType.RGB)
    imgwin = ImageWindow(image, window_width=512, window_height=512)
    return imgwin


def open_2Dsc():
    # opening a 2D single channel image:
    h = 5120
    w = 5120
    Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
    array = np.empty((h, w), dtype=np.float32)
    array[:] = np.random.rand(h, w)
    array[-30:] = np.linspace(0, 1, w)
    image = NImage(array, '2D1C', type=ImageType.Mono)
    imgwin = ImageWindow(image, window_width=512, window_height=512)
    imgwin.set_cmap("viridis")
    return imgwin


def open_3Dsc():
    # opening a 3D single channel image:
    h = 512
    w = 512
    d = 512
    Z, Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j, -2.5 : 2.5 : d * 1j]
    array = np.empty((h, w, d), dtype=np.float32)
    array[:] = np.exp(
        -X ** 2 - Y ** 2 - Z ** 2
    )  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    image = NImage(array, '3D1C', type=ImageType.Mono)
    imgwin = ImageWindow(image, window_width=512, window_height=512)
    imgwin.set_cmap("blues")
    return imgwin


def open_4dsc():
    # opening a 4D single channel image:
    h = 32
    w = 32
    d = 64
    b = 64
    C, Z, Y, X = np.ogrid[
        -2.5 : 2.5 : h * 1j,
        -2.5 : 2.5 : w * 1j,
        -2.5 : 2.5 : d * 1j,
        -2.5 : 2.5 : b * 1j,
    ]
    array = np.empty((h, w, d, b), dtype=np.float32)
    array[:] = np.exp(
        -X ** 2 - Y ** 2 - Z ** 2 - C ** 2
    )  # * (1. + .5*(np.random.rand(h, w)-.5))
    # image[-30:] = np.linspace(0, 1, w)
    image = NImage(array, '4D1C', type=ImageType.Mono)
    imgwin = ImageWindow(image)
    imgwin.set_cmap("blues")
    return imgwin


def open_emptymininap():
    array = np.empty((1, 1), dtype=np.float32)
    image = NImage(array)
    imgwin = ImageWindow(image)
    return imgwin


def update_image():
    h = 512
    w = 512
    Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
    array = np.empty((h, w), dtype=np.float32)
    array[:] = np.random.rand(h, w)
    array[-30:] = np.linspace(0, 1, w)
    image = NImage(array)
    imgwin = ImageWindow(image)
    h = 1112
    w = 1112
    Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
    array = np.empty((h, w), dtype=np.float32)
    array[:] = np.random.rand(h, w)
    array[-30:] = np.linspace(0, 1, w)
    image = NImage(array)
    imgwin.widget.image_canvas.image_visual.set_data(array)


if __name__ == '__main__':

    # starting
    application = NapariApplication(sys.argv)

    # imgwin1 = open_2Drgb()
    # imgwin2 = open_2Dsc()

    # imgwin3 = open_3Dsc()
    # imgwin4 = open_4dsc()
    # imgwin5 = open_emptymininap()

    update_image()

    sys.exit(application.exec_())
