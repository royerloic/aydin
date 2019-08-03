"""
Custom Qt widgets that serve as native objects that the public-facing elements
wrap.
"""
# set vispy to use same backend as qtpy
from qtpy import API_NAME
from vispy import app

app.use_app(API_NAME)
del app

from qtpy.QtWidgets import QMainWindow, QWidget, QHBoxLayout


class Window:
    """Application window that contains the menu bar and viewer.

    Parameters
    ----------
    qt_viewer : QtViewer
        Contained viewer widget.

    Attributes
    ----------
    qt_viewer : QtViewer
        Contained viewer widget.
    """

    def __init__(self, qt_viewer, *, show=False):

        self.qt_viewer = qt_viewer

        self._qt_window = QMainWindow()
        self._qt_window.setUnifiedTitleAndToolBarOnMac(True)
        self._qt_center = QWidget()
        self._qt_window.setCentralWidget(self._qt_center)
        self._qt_window.setWindowTitle(self.qt_viewer.viewer.title)
        self._qt_center.setLayout(QHBoxLayout())

        self._qt_center.layout().addWidget(self.qt_viewer)
        self._qt_center.layout().setContentsMargins(4, 0, 4, 0)

        self.qt_viewer.viewer.events.status.connect(self._status_changed)
        # self.qt_viewer.viewer.events.help.connect(self._help_changed)
        self.qt_viewer.viewer.events.title.connect(self._title_changed)

        if show:
            self.show()

    def resize(self, width, height):
        """Resize the window.

        Parameters
        ----------
        width : int
            Width in logical pixels.
        height : int
            Height in logical pixels.
        """
        self._qt_window.resize(width, height)

    def show(self):
        """Resize, show, and bring forward the window.
        """
        self._qt_window.resize(self._qt_window.layout().sizeHint())
        self._qt_window.show()
        self._qt_window.raise_()

    def _status_changed(self, event):
        """Update status bar.
        """
        # self._status_bar.showMessage(event.text)
        pass

    def _title_changed(self, event):
        """Update window title.
        """
        self._qt_window.setWindowTitle(event.text)
