import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QFrame


class AboutPage(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        self.lbl = QLabel()
        pixmap = QPixmap(os.path.abspath("aydin/gui/resources/biohub_logo.png"))
        pixmap = pixmap.scaled(313, 1000, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl.setPixmap(pixmap)
        self.layout.addWidget(self.lbl)

        # Horizontal Line Break
        self.hline_break = QFrame()
        self.hline_break.setFrameShape(QFrame.HLine)
        self.hline_break.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break)

        self.layout.addWidget(QLabel("Ahmet Can Solak"))
        self.layout.addWidget(QLabel("Hirofumi Kobayashi"))
        self.layout.addWidget(QLabel("Le Yan"))
        self.layout.addWidget(QLabel("Josh Batson"))
        self.layout.addWidget(QLabel("Loic Royer"))

        # Horizontal Line Break
        self.hline_break1 = QFrame()
        self.hline_break1.setFrameShape(QFrame.HLine)
        self.hline_break1.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break1)

        # Description
        self.layout.addWidget(QLabel("aydin - denoising but chill<br>"))

        self.repo_lbl = QLabel('<a href="http://czbiohub.org/">Source Code Repo</a>')
        self.repo_lbl.setOpenExternalLinks(True)
        self.layout.addWidget(self.repo_lbl)

        self.setLayout(self.layout)

        # TODO:  add description, add url to repo

    @staticmethod
    def showAbout():
        d = QDialog()
        d.setGeometry(150, 150, 350, 400)
        d.setFixedSize(350, 400)
        AboutPage(d)
        d.setWindowTitle("About")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
