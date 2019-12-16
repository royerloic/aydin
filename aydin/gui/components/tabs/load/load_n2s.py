from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGridLayout, QLineEdit, QDialog

from aydin.gui.components.filepath_picker import FilePathPicker
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.gui.components.util.format_warning import show_format_warning


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
        self.lbl_text = QLineEdit(self)
        self.lbl_text.hide()

        self.noisy_input_picker = FilePathPicker(
            self,
            "Noisy Image",
            self.input_lbl,
            self.image_info_lbl,
            self.lbl_text,
            file_ready=self.input_ready,
        )
        self.noisy_input_picker.setToolTip(
            "Click here to open file dialog or drag and drop your file to here to load your image."
        )
        paths_layout.addWidget(self.noisy_input_picker)

        input_image_layout = QGridLayout()

        paths_layout.addLayout(input_image_layout)

        self.base_layout.insertLayout(0, paths_layout)

    def on_input_ready(self):
        extentions = [".jpg", ".jpeg", ".jpe", ".jfif", ".jif", ".jfi"]
        if sum([ext in self.lbl_text.text() for ext in extentions]):
            self.image_info_lbl.hide()
            show_format_warning()
        else:
            self.wizard.next_tab()
