import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QLabel

from pitl.gui.components.mininap.gui.image_widget import ImageWidget
from pitl.gui.components.mininap.image.napari_image import NImage
from pitl.gui.components.tabs.base_tab import BaseTab


class ROIPage(BaseTab):
    def __init__(self, parent, image):
        super(ROIPage, self).__init__(parent)

        self.layout = QVBoxLayout()
        self.layout.addWidget(
            QLabel("!!!THIS IS WORK IN PROGRESS, DO NOT USE THIS TAB FOR NOW!!!")
        )
        # Setup mininap with passed image
        h = 5120
        w = 5120
        Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
        array = np.empty((h, w), dtype=np.float32)
        array[:] = np.random.rand(h, w)
        array[-30:] = np.linspace(0, 1, w)
        image = NImage(array)
        imgwin = ImageWidget(image)
        self.layout.addWidget(imgwin)

        # Add buttons to take snapshot of view
        self.snap_button = QPushButton("Snap")
        self.layout.addWidget(self.snap_button)

        self.base_layout.insertLayout(0, self.layout)
