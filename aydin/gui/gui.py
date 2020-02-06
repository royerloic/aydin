import atexit
import os
import platform
import sys

import click
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication
import qdarkstyle

from aydin.gui.pages.about import AboutPage
from aydin.gui.pages.start import StartPage
from aydin.util.update import download_specific_version, get_latest_version_details


class App(QMainWindow):
    def __init__(self, ver):
        super().__init__()

        self.version = ver

        self.threadpool = QThreadPool()

        self.title = 'aydin - Denoising, but chill...'
        self.left = 0
        self.top = 0
        self.width = 700 * 2 if platform.system() == "Windows" else 700
        self.height = 800 * 2 if platform.system() == "Windows" else 800
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.main_widget = StartPage(self, self.threadpool)
        self.setCentralWidget(self.main_widget)

        self.setupMenubar()

    def setupMenubar(self):
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu(' &File')
        helpMenu = mainMenu.addMenu(' &Help')

        # File Menu
        startPageButton = QAction('Start Page', self)
        startPageButton.setStatusTip('Go to start page')
        startPageButton.triggered.connect(
            lambda: self.setCentralWidget(StartPage(self, self.threadpool))
        )
        fileMenu.addAction(startPageButton)

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Help Menu
        versionButton = QAction("ver" + self.version, self)
        helpMenu.addAction(versionButton)
        aboutButton = QAction('About', self)
        aboutButton.setStatusTip('About aydin and its authors')
        aboutButton.triggered.connect(AboutPage.showAbout)
        helpMenu.addAction(aboutButton)

        updateButton = QAction('Update', self)
        updateButton.setStatusTip('Check updates and apply if there is any')
        updateButton.triggered.connect(self.update_app)
        helpMenu.addAction(updateButton)

    def setHeight(self, height):
        self.height = height
        self.setGeometry(self.left, self.top, self.width, self.height)

    def update_app(self):
        # Print out current version
        print("this is a test version3 ", self.version)

        # Check updates and download if there is
        latest_version, latest_id = get_latest_version_details()

        if latest_version > self.version:
            print(
                "There is a more recent version of Aydin, automatically updating and re-running now..."
            )
            # Download new version
            path_to_new_version = download_specific_version(latest_version, latest_id)

            # Run new version with same command and args
            args = click.get_os_args()
            words = [path_to_new_version] + args
            path_to_run = ' '.join(words)

            atexit.register(lambda: os.system(path_to_run))
            self.close()
        else:
            print("You are running the most updated version of Aydin")


def run(ver):
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App(ver)
    ex.show()
    sys.exit(app.exec())
