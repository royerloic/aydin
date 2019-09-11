from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout

from aydin.gui.components.filepath_picker import FilePathPicker
from aydin.gui.components.tabs.base_tab import BaseTab


class UploadN2STab(BaseTab):
    input_ready = pyqtSignal()

    def __init__(self, parent):
        super(UploadN2STab, self).__init__(parent)
        self.input_ready.connect(self.on_input_ready)

        """
        Paths layout where we list required paths and what are current values for those
        Also, these boxes are drag-and-drop areas. User drag-and-drop any file or folder,
        or user can set the path with the help of button on the right end.
        """
        paths_layout = QVBoxLayout()
        paths_layout.addWidget(QLabel("Path for the input training noisy images: "))
        self.input_lbl = QLabel(self)
        self.input_picker = FilePathPicker(self, self.input_lbl, self.input_ready)
        paths_layout.addWidget(self.input_picker)

        pixmaps_layout = QHBoxLayout()
        pixmaps_layout.addWidget(self.input_lbl)
        paths_layout.addLayout(pixmaps_layout)

        self.base_layout.insertLayout(0, paths_layout)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def on_input_ready(self):
        self.next_button.setEnabled(True)
