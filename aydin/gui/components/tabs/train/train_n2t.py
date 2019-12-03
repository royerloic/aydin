import numpy as np
from skimage.io import imsave
from PyQt5.QtWidgets import QVBoxLayout, QPushButton, QProgressBar, QHBoxLayout

from aydin.gui.components.mininap import Viewer
from aydin.gui.components.plot_canvas import PlotCanvas
from aydin.gui.components.step_progress_bar import StepProgressBar
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.gui.components.workers.worker import Worker
from aydin.services.n2t import N2TService
from aydin.util.resource import read_image_from_path


class TrainN2TTab(BaseTab):

    is_loaded = False

    def __init__(self, parent, threadpool):
        super(TrainN2TTab, self).__init__(parent)

        self.wizard = parent
        self.threadpool = threadpool
        self.layout = None

        self.noisy_input_picker = self.wizard.upload_tab.noisy_input_picker
        self.truth_input_picker = self.wizard.upload_tab.truth_input_picker
        self.test_input_picker = self.wizard.upload_tab.test_input_picker

        self.n2t = N2TService()

    def load_tab(self):
        self.setGeometry(0, 0, 700, 800)
        self.layout = QVBoxLayout()

        # Buttons layout where we have run button and other functional methods
        tab_layout = QVBoxLayout()

        self.progress_bar = StepProgressBar()
        tab_layout.addWidget(self.progress_bar)

        self.pb = PlotCanvas(self)
        tab_layout.addWidget(self.pb)

        self.viewer = Viewer(show=False)
        self.viewer.add_image(self.wizard.monitor_images[0][np.newaxis, ...])
        tab_layout.addWidget(self.viewer.window.qt_viewer)

        self.final_button_layout = QHBoxLayout()

        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setToolTip("Stop currently running Noise2Truth training")
        self.stop_button.pressed.connect(self.n2t.stop_func)
        self.stop_button.setDisabled(True)
        self.final_button_layout.addWidget(self.stop_button)

        self.savemodel_button = QPushButton("Save the model")
        self.savemodel_button.setToolTip("Save the trained model to infer on later")
        self.savemodel_button.setEnabled(False)
        self.final_button_layout.addWidget(self.savemodel_button)

        self.run_button = QPushButton("Start Training")
        self.run_button.setToolTip(
            "Start running Noise2Truth training with selected input and options"
        )
        self.run_button.pressed.connect(
            lambda: Worker.enqueue_funcname(
                self.threadpool, self.run_func, self.progressbar_update
            )
        )

        tab_layout.addLayout(self.final_button_layout)

        # Add into main layout
        self.layout = tab_layout
        self.base_layout.insertLayout(0, self.layout)

        self.is_loaded = True
        self.run_button.click()

    def toggle_button_availablity(self):
        self.run_button.setDisabled(self.run_button.isEnabled())
        self.stop_button.setDisabled(self.stop_button.isEnabled())

    def progressbar_update(self, value):
        if 0 <= value <= 100:
            self.progress_bar.emit(value)

        if value == 100:
            self.toggle_button_availablity()  # Toggle buttons back by the end of the run

    def run_func(self, **kwargs):
        self.toggle_button_availablity()  # Toggle buttons to prevent multiple run actions and so

        noisy_path = self.noisy_input_picker.lbl_text.text()
        noisy = read_image_from_path(noisy_path)
        truth_path = self.truth_input_picker.lbl_text.text()
        truth = read_image_from_path(truth_path)
        test_path = self.test_input_picker.lbl_text.text()
        noisy_test = read_image_from_path(test_path)

        output_path = test_path[:-4] + "_denoised" + test_path[-4:]

        print(self.wizard.monitor_images)

        denoised = self.n2t.run(
            noisy,
            truth,
            noisy_test,
            kwargs['progress_callback'],
            monitoring_callbacks=[self.update_test_tab],
            monitoring_images=self.wizard.monitor_images,
        )

        imsave(output_path, denoised)
        self.run_button.setText("Re-Run")
        print(output_path)
        return "Done."

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
