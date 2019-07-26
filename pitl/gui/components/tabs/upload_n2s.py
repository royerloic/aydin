from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout

from pitl.gui.components.filepath_picker import FilePathPicker
from pitl.gui.components.tabs.base_tab import BaseTab


class UpdateN2STab(BaseTab):
    def __init__(self, parent):
        super(UpdateN2STab, self).__init__(parent)

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

        self.base_layout.insertLayout(0, paths_layout)
