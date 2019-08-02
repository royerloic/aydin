import numpy as np
from skimage.io import imsave
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QPlainTextEdit,
)

from aydin.gui.components.mininap.gui.image_widget import ImageWidget
from aydin.gui.components.mininap.image.napari_image import NImage
from aydin.gui.components.plot_canvas import PlotCanvas
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.gui.components.workers.worker import Worker
from aydin.services.n2s import N2SService
from aydin.util.resource import read_image_from_path


class TestN2STab(BaseTab):
    def __init__(self, parent, threadpool):
        super(TestN2STab, self).__init__(parent)

        self.wizard = parent
        self.threadpool = threadpool
        self.layout = None

        self.input_picker = self.wizard.upload_tab.input_picker
        self.output_picker = self.wizard.upload_tab.output_picker

    def load_tab(self):
        self.setGeometry(0, 0, 700, 800)
        self.layout = QVBoxLayout()

        # Buttons layout where we have run button and other functional methods
        buttons_layout = QVBoxLayout()
        self.pb = PlotCanvas(self)
        buttons_layout.addWidget(self.pb.canvas.native)

        self.run_button = QPushButton("Run")
        self.run_button.pressed.connect(
            lambda: Worker.enqueue_funcname(
                self.threadpool, self.run_func, self.progressbar_update
            )
        )
        buttons_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar(self)
        buttons_layout.addWidget(self.progress_bar)

        self.image_data = self.wizard.monitor_image
        self.image = NImage(self.image_data)
        self.imgwid = ImageWidget(self.image)
        buttons_layout.addWidget(self.imgwid)

        # Build splitter
        def_splitter = QSplitter(Qt.Vertical)

        paths_and_buttons = QWidget()
        paths_and_buttons.setLayout(buttons_layout)
        def_splitter.addWidget(paths_and_buttons)

        # Add splitter into main layout
        self.layout.addWidget(def_splitter)
        self.base_layout.insertLayout(0, self.layout)

    def progressbar_update(self, value):
        if 0 <= value <= 100:
            self.pb.add_pos(100 + value)
            self.progress_bar.setValue(value)

    def run_func(self, **kwargs):

        input_path = self.input_picker.lbl_text.text()
        noisy = read_image_from_path(input_path)

        output_path = self.output_picker.lbl_text.text()
        if len(output_path) <= 0:
            output_path = input_path[:-4] + "_denoised" + input_path[-4:]
            self.output_picker.lbl_text.setText(output_path)

        n2s = N2SService(monitoring_variables_emit=self.add_to_mininap)
        n2s.monitoring_variables = None, 0, -1

        denoised = n2s.run(
            noisy,
            kwargs['progress_callback'],
            monitoring_images=[self.wizard.monitor_image],
        )

        imsave(output_path, denoised)
        self.run_button.setText("Re-Run")
        self.output_picker.filename = output_path
        self.output_picker.load_file()
        print(output_path)
        return "Done."

    def add_to_mininap(self, arg):
        image, eval_metric, iter = arg
        self.image_data = np.dstack(
            (self.image_data, image[0])
        )  # Turn this into slider forming
        self.imgwid.update_image(NImage(self.image_data))
