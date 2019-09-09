from PyQt5.Qt import *
import sys
import os


class FilePathPicker(QWidget):
    """
    Subclass the widget and add a button to load images.
    Alternatively set up dragging and dropping of image files onto the widget
    """

    def __init__(self, parent, lbl, file_ready=None):
        super(FilePathPicker, self).__init__(parent)
        self.parent = parent
        self.file_ready = file_ready
        self.filename = None

        # Button that allows loading of images
        self.load_button = QPushButton("Load file path \n (Drag and drop here...)")
        self.load_button.setMinimumSize(120, 120)
        self.load_button.clicked.connect(self.load_file_button)

        # Path viewing region
        self.lbl_text = QLineEdit(self)
        self.lbl = lbl

        # A horizontal layout to include the button on the left
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.load_button)
        layout_button.addWidget(self.lbl_text)

        # A Vertical layout to include the button layout and then the image
        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        # layout.addWidget(self.lbl)

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
            self.lbl_text.setText(self.filename)
            pixmap = QPixmap(self.filename)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
            self.lbl.setPixmap(pixmap)
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
