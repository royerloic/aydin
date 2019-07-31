import sys

from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QSplitter
import qdarkstyle

from pitl.gui.pages.about import AboutPage
from pitl.gui.pages.welcome import WelcomePage


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

        self.main_widget = WelcomePage(self, self.threadpool)
        self.setCentralWidget(self.main_widget)

        self.setupMenubar()

    def setupMenubar(self):
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu(' &File')
        helpMenu = mainMenu.addMenu(' &Help')

        # File Menu
        welcomePageButton = QAction('Welcome Page', self)
        welcomePageButton.setStatusTip('Go to welcome page')
        welcomePageButton.triggered.connect(
            lambda: self.setCentralWidget(WelcomePage(self, self.threadpool))
        )
        fileMenu.addAction(welcomePageButton)

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Help Menu
        aboutButton = QAction('About', self)
        aboutButton.setStatusTip('About pitl and its authors')
        aboutButton.triggered.connect(AboutPage.showAbout)
        helpMenu.addAction(aboutButton)

    def setHeight(self, height):
        self.height = height
        self.setGeometry(self.left, self.top, self.width, self.height)


def run():
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ex = App()
    ex.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
