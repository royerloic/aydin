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

from pitl.gui.components.mininap.gui.image_widget import ImageWidget
from pitl.gui.components.mininap.image.napari_image import NImage
from pitl.gui.components.plot_canvas import PlotCanvas
from pitl.gui.components.tabs.base_tab import BaseTab
from pitl.gui.components.workers.worker import Worker
from pitl.services.Noise2Self import Noise2Self
from pitl.util.resource import read_image_from_path


class TestN2STab(BaseTab):
    def __init__(self, parent, threadpool):
        super(TestN2STab, self).__init__(parent)
        self.wizard = parent

        self.setGeometry(0, 0, 700, 800)

        self.threadpool = threadpool

        self.inputfile_picker = self.wizard.upload_tab.inputfile_picker
        self.outputfile_picker = self.wizard.upload_tab.outputfile_picker

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

        h = 512
        w = 512
        Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
        array = np.empty((h, w), dtype=np.float32)
        array[:] = np.random.rand(h, w)
        array[-30:] = np.linspace(0, 1, w)
        self.image = NImage(array)
        self.imgwin = ImageWidget(self.image)
        buttons_layout.addWidget(self.imgwin)

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

        input_path = self.inputfile_picker.lbl_text.text()
        noisy = read_image_from_path(input_path)

        output_path = self.outputfile_picker.lbl_text.text()
        if len(output_path) <= 0:
            output_path = input_path[:-4] + "_denoised" + input_path[-4:]
            self.outputfile_picker.lbl_text.setText(output_path)

        denoised = Noise2Self.run(noisy, kwargs['progress_callback'])
        # self.imgwin.setParent(None)
        self.imgwin.update_image(
            NImage(denoised)
        )  # THIS IS WORKING BUT CMAP IS NOT NORMAL
        self.imgwin.set_cmap = 'gray'
        # self.image = NImage(denoised)
        # self.imgwin = ImageWidget(self.image)

        imsave(output_path, denoised)
        self.run_button.setText("Re-Run")
        self.outputfile_picker.filename = output_path
        self.outputfile_picker.load_file()
        print(output_path)
        return "Done."
