import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel
from vispy.visuals.transforms import STTransform

from aydin.gui.components.mininap import Viewer
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.util.resource import read_image_from_path


class ROITab(BaseTab):
    roi_ready = pyqtSignal()
    is_loaded = False

    def __init__(self, parent):
        super(ROITab, self).__init__(parent)
        self.roi_ready.connect(self.on_roi_ready)

        self.wizard = parent
        self.image = None
        self.stt = STTransform()

    def load_tab(self):
        # Read the input image
        self.image_data = read_image_from_path(
            self.wizard.upload_noisy_tab.input_picker.lbl_text.text()
        )

        self.layout = QVBoxLayout()

        # Friendly explanation
        self.layout.addWidget(
            QLabel(
                "Please select the ROI with help of viewer below or program will continue with the default selection."
            )
        )

        # Setup mininap with passed image
        self.viewer = Viewer(show=False)
        self.viewer.add_image(self.image_data)
        self.layout.addWidget(self.viewer.window.qt_viewer)

        # Add buttons to take snapshot of view
        self.snap_button = QPushButton("Set ROI")
        self.snap_button.setToolTip("Select Region-Of-Interest that would be monitored")
        self.snap_button.pressed.connect(self.snap_test)
        self.layout.addWidget(self.snap_button)

        self.base_layout.insertLayout(0, self.layout)

        self.next_button.setEnabled(False)
        self.is_loaded = True

    def snap_test(self):
        # Get the image size and visual and canvas size
        image_size = tuple(
            np.squeeze(
                self.viewer.window.qt_viewer.view.camera.viewbox.get_scene_bounds()[:2]
            )[:, 1]
        )
        transform = self.viewer.layers[0]._node.canvas.scene.node_transform(
            self.viewer.layers[0]._node
        )
        w, h = self.viewer.window.qt_viewer.canvas.size

        # compute distance of image to top left and bottom right corners
        to_top_left, to_bottom_right = (
            transform.map([0, 0])[:2],
            transform.map([w, h])[:2],
        )
        scaled_canvas_width, scaled_canvas_height = (
            -1 * (to_top_left[0]) + to_bottom_right[0],
            -1 * (to_top_left[1]) + to_bottom_right[1],
        )

        # l1,r1 for visible canvas region, l2,r2 for our image
        l1 = 0, 0
        r1 = scaled_canvas_width, scaled_canvas_height
        l2 = -1 * to_top_left
        r2 = l2 + image_size

        # Check if image is in scene
        if l1[0] > r2[0] or l2[0] > r1[0]:
            print("Out of sight, f")
            return

        if l1[1] > r2[1] or l2[1] > r1[1]:
            print("Out of sight, s")
            return

        # Find the intersection of image and visible canvas
        l3 = max(l1[0], l2[0]), max(l1[1], l2[1])
        r3 = min(r1[0], r2[0]), min(r1[1], r2[1])

        # Get indices for slicing image
        p1, p2 = list(l3) - l2, list(r3) - l2
        print(p1, p2)

        # Slice image and replace
        self.wizard.monitor_images.append(
            self.image_data[int(p1[0]) : int(p2[0]), int(p1[1]) : int(p2[1])]
        )

        self.roi_ready.emit()

    def on_roi_ready(self):
        self.next_button.setEnabled(True)
