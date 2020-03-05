from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QDialog


def show_format_warning():
    d = QDialog()
    d.setGeometry(150, 150, 350, 400)
    d.setFixedSize(500, 50)
    lbl = QLabel(
        "JPEG format images are not supported. Please use a lossless image format.", d
    )
    lbl.setEnabled(True)
    d.setWindowTitle("Format Warning")
    d.setWindowModality(Qt.ApplicationModal)
    d.exec_()
