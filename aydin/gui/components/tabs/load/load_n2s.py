from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QHBoxLayout, QGridLayout

from aydin.gui.components.filepath_picker import FilePathPicker
from aydin.gui.components.tabs.base_tab import BaseTab


class LoadN2STab(BaseTab):
    input_ready = pyqtSignal()

    def __init__(self, parent):
        super(LoadN2STab, self).__init__(parent)
        self.input_ready.connect(self.on_input_ready)

        """
        Paths layout where we list required paths and what are current values for those
        Also, these boxes are drag-and-drop areas. User drag-and-drop any file or folder,
        or user can set the path with the help of button on the right end.
        """
        paths_layout = QVBoxLayout()

        self.input_lbl = QLabel(self)
        self.image_info_lbl = QLabel(self)
        self.input_picker = FilePathPicker(
            self, self.input_lbl, self.image_info_lbl, self.input_ready
        )
        self.input_picker.setToolTip(
            "Click here to open file dialog or drag and drop your file to here to load your image."
        )
        paths_layout.addWidget(self.input_picker)

        input_image_layout = QGridLayout()
        input_image_layout.addWidget(self.image_info_lbl)

        paths_layout.addLayout(input_image_layout)

        self.base_layout.insertLayout(0, paths_layout)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def on_input_ready(self):
        self.next_button.setEnabled(True)
