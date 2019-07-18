from PyQt5.QtCore import QThreadPool, pyqtSignal, QRunnable, QObject, pyqtSlot, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QPlainTextEdit,
    QApplication,
)

from pitl.gui.components.filepath_picker import FilePathPicker
from pitl.gui.components.plot_canvas import PlotCanvas
from pitl.gui.components.worker import Worker
from pitl.services.Noise2Self import Noise2Self
from pitl.util.resource import read_image_from_path
from skimage.io import imsave
from pitl.gui.components.dlineplotexample import MyCanvas


class Noise2SelfTab(QWidget):
    def __init__(self, parent, threadpool):
        super(QWidget, self).__init__(parent)

        self.threadpool = threadpool

        self.layout = QVBoxLayout()

        """
        Paths layout where we list required paths and what are current values for those
        Also, these boxes are drag-and-drop areas. User drag-and-drop any file or folder,
        or user can set the path with the help of button on the right end.
        """
        paths_layout = QVBoxLayout()
        paths_layout.addWidget(QLabel("Path for the input training noisy images: "))
        self.input_lbl = QLabel(self)
        self.inputfile_picker = FilePathPicker(self.input_lbl)
        paths_layout.addWidget(self.inputfile_picker)
        paths_layout.addWidget(QLabel("Path to save resulting denoised images: "))
        self.output_lbl = QLabel(self)
        self.outputfile_picker = FilePathPicker(self.output_lbl)
        paths_layout.addWidget(self.outputfile_picker)
        pixmaps_layout = QHBoxLayout()
        pixmaps_layout.addWidget(self.input_lbl)
        pixmaps_layout.addWidget(self.output_lbl)
        paths_layout.addLayout(pixmaps_layout)

        # Buttons layout where we have run button and other functional methods
        buttons_layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.pressed.connect(
            lambda: Worker.enqueue_funcname(
                self.threadpool, self.run_func, self.progressbar_update
            )
        )
        buttons_layout.addWidget(self.run_button)

        # Build splitter
        def_splitter = QSplitter(Qt.Vertical)

        def_splitter.addWidget(QPlainTextEdit())  # for tab definitions

        paths_and_buttons_layout = QVBoxLayout()
        paths_and_buttons_layout.addLayout(paths_layout)
        paths_and_buttons_layout.addLayout(buttons_layout)
        self.progress_bar = QProgressBar(self)
        paths_and_buttons_layout.addWidget(self.progress_bar)
        self.pb = PlotCanvas(self)
        paths_and_buttons_layout.addWidget(self.pb.canvas.native)

        paths_and_buttons = QWidget()
        paths_and_buttons.setLayout(paths_and_buttons_layout)
        def_splitter.addWidget(paths_and_buttons)

        # Add splitter into main layout
        self.layout.addWidget(def_splitter)
        self.setLayout(self.layout)

    def progressbar_update(self, value):
        if 0 <= value <= 100:
            self.pb.add_pos(100 + value)
            self.progress_bar.setValue(value)

    def run_func(self, progress_callback):
        self.run_button.setStyleSheet("background-color: orange")

        input_path = self.inputfile_picker.lbl_text.text()
        noisy = read_image_from_path(input_path)

        output_path = self.outputfile_picker.lbl_text.text()
        if len(output_path) <= 0:
            output_path = input_path[:-4] + "_denoised" + input_path[-4:]
            self.outputfile_picker.lbl_text.setText(output_path)

        denoised = Noise2Self.run(noisy, progress_callback)

        imsave(output_path, denoised)
        self.run_button.setText("Re-Run")
        self.run_button.setStyleSheet("background-color: green")
        self.outputfile_picker.filename = output_path
        self.outputfile_picker.load_file()
        print(output_path)
        return "Done."
