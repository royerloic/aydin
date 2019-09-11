from time import sleep

import numpy as np
from skimage.io import imsave
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QProgressBar,
    QSplitter,
    QHBoxLayout,
)

from aydin.gui.components.mininap import Viewer
from aydin.gui.components.plot_canvas import PlotCanvas
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.gui.components.workers.worker import Worker
from aydin.services.n2s import N2SService
from aydin.util.resource import read_image_from_path


class TestN2STab(BaseTab):

    is_loaded = False

    def __init__(self, parent, threadpool):
        super(TestN2STab, self).__init__(parent)

        self.wizard = parent
        self.threadpool = threadpool
        self.layout = None

        self.input_picker = self.wizard.upload_tab.input_picker

        self.n2s = N2SService()

    def load_tab(self):
        self.setGeometry(0, 0, 700, 800)
        self.layout = QVBoxLayout()

        # Buttons layout where we have run button and other functional methods
        tab_layout = QVBoxLayout()
        self.pb = PlotCanvas(self)
        tab_layout.addWidget(self.pb)

        self.stop_button = QPushButton("Stop")
        self.stop_button.pressed.connect(self.stop_func)
        self.stop_button.setDisabled(True)

        self.run_button = QPushButton("Run")
        self.run_button.pressed.connect(
            lambda: Worker.enqueue_funcname(
                self.threadpool, self.run_func, self.progressbar_update
            )
        )

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.run_button)
        tab_layout.addLayout(buttons_layout)

        self.progress_bar = QProgressBar(self)
        tab_layout.addWidget(self.progress_bar)

        self.viewer = Viewer(show=False)
        self.viewer.add_image(self.wizard.monitor_images[0][np.newaxis, ...])
        tab_layout.addWidget(self.viewer.window.qt_viewer)

        # Build splitter
        def_splitter = QSplitter(Qt.Vertical)

        test_n2s_tab_view = QWidget()
        test_n2s_tab_view.setLayout(tab_layout)
        def_splitter.addWidget(test_n2s_tab_view)

        # Add splitter into main layout
        self.layout.addWidget(def_splitter)
        self.base_layout.insertLayout(0, self.layout)

        self.is_loaded = True

    def progressbar_update(self, value):
        if 0 <= value <= 100:
            self.progress_bar.setValue(value)

        if value == 100:
            # TODO: refactor this into a method with check isEnabled()
            self.stop_button.setDisabled(True)
            self.run_button.setDisabled(False)
            self.prev_button.setDisabled(False)
            self.next_button.setDisabled(False)
            self.wizard.setTabEnabled(0, True)
            self.wizard.setTabEnabled(1, True)

    def run_func(self, **kwargs):
        # TODO: refactor this into a method with check isEnabled()
        self.wizard.setTabEnabled(0, False)
        self.wizard.setTabEnabled(1, False)
        self.prev_button.setDisabled(True)
        self.next_button.setDisabled(True)
        self.run_button.setDisabled(True)
        self.stop_button.setDisabled(False)

        input_path = self.input_picker.lbl_text.text()
        noisy = read_image_from_path(input_path)

        output_path = input_path[:-4] + "_denoised" + input_path[-4:]

        print(self.wizard.monitor_images)

        denoised = self.n2s.run(
            noisy,
            kwargs['progress_callback'],
            monitoring_callbacks=[self.update_test_tab],
            monitoring_images=self.wizard.monitor_images,
        )

        imsave(output_path, denoised)
        self.run_button.setText("Re-Run")
        print(output_path)
        return "Done."

    def stop_func(self):
        self.n2s.it.stop_training()

    def update_test_tab(self, *arg):
        """
        This is the function that is evoked as a callback to update UI
        with results of most updated model.

        :param arg:
        :return:
        """
        iter, eval_metric, image = arg[0]  # Parse callback arguments
        image = image[0][
            np.newaxis, ...
        ]  # Reshape 2D inferred image to 3D for stacking

        self.pb.add_val(eval_metric)

        if image is not None:
            keep_slider_location = False
            if self.viewer.dims.point[0] == self.viewer.layers[0].data.shape[0] - 1:
                keep_slider_location = True

            self.viewer.layers[0].data = np.vstack((self.viewer.layers[0].data, image))
            self.viewer.window.qt_viewer.dims._update_slider(0)

            if keep_slider_location:
                self.viewer.dims.set_point(0, self.viewer.layers[0].data.shape[0] - 1)
