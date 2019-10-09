import os

from PyQt5.Qt import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, width, height):
        super(ClickableLabel, self).__init__()
        self.pixmapo = QPixmap(
            os.path.abspath("aydin/gui/resources/draganddroplogo.png")
        )
        self.pixmapo = self.pixmapo.scaled(700, 500, Qt.KeepAspectRatio)
        self.setPixmap(self.pixmapo)

    def mousePressEvent(self, event):
        self.clicked.emit()

    def changeView(self, filename):
        self.pixmapo = QPixmap(QPixmap(filename))
        self.pixmapo = self.pixmapo.scaled(700, 500, Qt.KeepAspectRatio)
        self.setPixmap(self.pixmapo)
