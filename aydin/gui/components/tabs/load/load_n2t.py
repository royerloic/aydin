from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QDialog,
)

from aydin.gui.components.filepath_picker import FilePathPicker
from aydin.gui.components.tabs.base_tab import BaseTab
from aydin.gui.components.util.format_warning import show_format_warning


class LoadN2TTab(BaseTab):
    input_ready = pyqtSignal()

    noisy_ready = False
    truth_ready = False
    test_ready = False

    def __init__(self, parent):
        super(LoadN2TTab, self).__init__(parent)
        self.input_ready.connect(self.on_all_ready)

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

        noisy_and_truth_row_layout = QHBoxLayout()

        # Noisy image input
        self.noisy_input_picker = FilePathPicker(
            self,
            "Noisy Image",
            self.input_lbl,
            self.image_info_lbl,
            self.lbl_text,
            max_height=200,
            max_width=200,
            file_ready=self.input_ready,
        )
        self.noisy_input_picker.setToolTip(
            "Click here to open file dialog or drag and drop your file to here to load your image."
        )
        noisy_and_truth_row_layout.addWidget(self.noisy_input_picker)

        self.arrow_lbl = QLabel()
        self.arrow_pixmap = QPixmap("aydin/gui/resources/horizontal_arrow.png")
        self.arrow_pixmap = self.arrow_pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        self.arrow_lbl.setPixmap(self.arrow_pixmap)
        noisy_and_truth_row_layout.addWidget(self.arrow_lbl)

        # Truth image input
        self.truth_input_picker = FilePathPicker(
            self,
            "Truth Image",
            self.input_lbl,
            self.image_info_lbl,
            self.lbl_text,
            max_height=200,
            max_width=200,
            file_ready=self.input_ready,
        )
        self.truth_input_picker.setToolTip(
            "Click here to open file dialog or drag and drop your file to here to load your image."
        )
        noisy_and_truth_row_layout.addWidget(self.truth_input_picker)

        # add to main paths layout
        paths_layout.addLayout(noisy_and_truth_row_layout)

        # Test image input
        self.test_input_picker = FilePathPicker(
            self,
            "Test Image",
            self.input_lbl,
            self.image_info_lbl,
            self.lbl_text,
            max_height=200,
            max_width=200,
            file_ready=self.input_ready,
        )
        self.test_input_picker.setToolTip(
            "Click here to open file dialog or drag and drop your file to here to load your image."
        )
        paths_layout.addWidget(self.test_input_picker)

        self.ready_button = QPushButton("Ready")
        self.ready_button.pressed.connect(self.wizard.next_tab)
        self.ready_button.setEnabled(False)
        paths_layout.addWidget(self.ready_button)

        self.base_layout.insertLayout(0, paths_layout)

    def on_all_ready(self):
        extentions = [".jpg", ".jpeg", ".jpe", ".jfif", ".jif", ".jfi"]
        if sum([ext in self.lbl_text.text() for ext in extentions]):
            self.image_info_lbl.hide()
            show_format_warning()
        else:
            if (
                self.noisy_input_picker.ready_flag
                and self.truth_input_picker.ready_flag
                and self.test_input_picker.ready_flag
            ):
                self.ready_button.setEnabled(True)
