import sys
import numpy as np

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QSplitter
import qdarkstyle

from pitl.gui.components.log_console import LogConsole
from pitl.gui.mininap.gui.image_widget import ImageWidget
from pitl.gui.mininap.image.napari_image import NImage
from pitl.gui.tabs.tabs import Tabs


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.threadpool = QThreadPool()

        self.title = 'Cool Image Translation'
        self.left = 0
        self.top = 0
        self.width = 700
        self.height = 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.tabs = Tabs(self, self.threadpool)
        # self.log_console = LogConsole(self)
        self.main_widget = QSplitter(Qt.Horizontal)
        self.control_widget = QSplitter(Qt.Vertical)
        self.control_widget.addWidget(self.tabs)
        # self.control_widget.addWidget(self.log_console)
        self.control_widget.setSizes([1, 0])

        self.main_widget.addWidget(self.control_widget)

        # trial mininap
        h = 5120
        w = 5120
        Y, X = np.ogrid[-2.5 : 2.5 : h * 1j, -2.5 : 2.5 : w * 1j]
        array = np.empty((h, w), dtype=np.float32)
        array[:] = np.random.rand(h, w)
        array[-30:] = np.linspace(0, 1, w)
        image = NImage(array)
        imgwin = ImageWidget(image)
        self.main_widget.addWidget(imgwin)
        self.setCentralWidget(self.main_widget)

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu(' &File')
        searchMenu = mainMenu.addMenu(' &Search')
        helpMenu = mainMenu.addMenu(' &Help')

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)


def run():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
