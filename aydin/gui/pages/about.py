from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog


class AboutPage(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        self.layout = QVBoxLayout()

        self.layout.addWidget(QLabel("Ahmet Can Solak"))
        self.layout.addWidget(QLabel("Hirofumi Kobayashi"))
        self.layout.addWidget(QLabel("Le Yan"))
        self.layout.addWidget(QLabel("Josh Batson"))
        self.layout.addWidget(QLabel("Loic Royer"))

        self.setLayout(self.layout)

    @staticmethod
    def showAbout(self):
        d = QDialog()
        d.setGeometry(150, 150, 200, 200)
        AboutPage(d)
        d.setWindowTitle("About")
        d.setWindowModality(Qt.ApplicationModal)
        d.exec_()
