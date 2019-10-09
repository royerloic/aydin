from PyQt5.Qt import *
import sys
import os

from aydin.gui.components.clickable_label import ClickableLabel
from aydin.io import imread


class FilePathPicker(QWidget):
    """
    Subclass the widget and add a button to load images.
    Alternatively set up dragging and dropping of image files onto the widget
    """

    def __init__(self, parent, img_lbl, info_lbl, file_ready=None):
        super(FilePathPicker, self).__init__(parent)
        self.parent = parent
        self.file_ready = file_ready
        self.filename = None

        # Button that allows loading of images
        # self.load_button.setMinimumSize(40, 40)
        self.load_button = ClickableLabel(700, 500)
        self.load_button.setMaximumSize(700, 500)

        self.load_button.clicked.connect(self.load_file_button)

        # Path viewing region
        self.lbl_text = QLineEdit(self)
        self.lbl = img_lbl
        self.info_lbl = info_lbl

        # A horizontal layout to include the button on the left
        layout_button = QVBoxLayout()
        layout_button.addWidget(self.load_button)
        layout_button.addWidget(self.lbl_text)

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)

        self.setLayout(layout)

        # Enable dragging and dropping onto the GUI
        self.setAcceptDrops(True)

        self.show()

    def load_file_button(self):
        """
        Open a File dialog when the button is pressed

        :return:
        """

        # Get the file location

        self.filename, _ = QFileDialog.getOpenFileName(QFileDialog(), 'Open file')
        # Load the image from the location
        self.load_file()

    def load_file(self):
        """
        Set the fname to label

        :return:
        """
        if os.path.isfile(self.filename):

            # path holder
            self.lbl_text.setText(self.filename)

            # img
            self.load_button.changeView(self.filename)

            # info
            _, metadata = imread(self.filename)
            to_print = [
                "ext : " + str(metadata.extension),
                "axes : " + str(metadata.axes),
                "dtype : " + str(metadata.dtype),
                "shape : " + str(metadata.shape),
            ]
            string2print = "\n".join(to_print)
            self.info_lbl.setText(string2print)

            if (
                self.file_ready is not None
            ):  # TODO: check if this if needed once everything connected on UI
                self.file_ready.emit()
        else:
            raise Exception("Selected item is not a file...")

    # The following three methods set up dragging and dropping for the app
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        Drop files directly onto the widget
        File locations are stored in fname

        :param e:
        :return:
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()

            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())

            self.filename = fname
            self.load_file()
        else:
            e.ignore()


# Demo, Runs if called directly
if __name__ == '__main__':
    # Initialise the application
    app = QApplication(sys.argv)
    # Call the widget
    ex = FilePathPicker()
    sys.exit(app.exec_())
