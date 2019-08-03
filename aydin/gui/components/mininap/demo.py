import sys

import numpy as np
from PyQt5.QtWidgets import QApplication
from vispy import app
from aydin.gui.components.mininap.viewer import Viewer

app.use_app('pyqt5')


def update_image():
    h = 512
    w = 512
    d = 512
    Z, Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j, -2.5 : 2.5 : d * 1j]
    array = np.empty((h, w, d), dtype=np.float32)
    array[:] = np.exp(-X ** 2 - Y ** 2 - Z ** 2)

    # create the viewer with an image
    viewer = Viewer()
    layer = viewer.add_image(array, title='napari example')

    # adjust some of the layer properties
    layer.opacity = 0.9
    layer.blending = 'translucent'
    layer.colormap = 'gray'
    layer.interpolation = 'nearest'

    array = np.vstack((array, array))
    layer.data = array


if __name__ == '__main__':
    # starting
    application = QApplication(sys.argv)

    update_image()

    sys.exit(application.exec_())
