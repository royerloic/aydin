from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel

from aydin.util.resource import resource_path


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent, height, width):
        super(ClickableLabel, self).__init__(parent)
        self.parent = parent

        self.height = height
        self.width = width

        self.pixmap = QPixmap(resource_path("aydin/gui/resources/draganddroplogo.png"))
        self.pixmap = self.pixmap.scaled(height, width, Qt.KeepAspectRatio)
        self.setPixmap(self.pixmap)

    def mousePressEvent(self, event):
        self.clicked.emit()

    def changeView(self, filename):
        self.pixmap = QPixmap(QPixmap(filename))
        self.pixmap = self.pixmap.scaled(self.height, self.width, Qt.KeepAspectRatio)
        self.setPixmap(self.pixmap)
