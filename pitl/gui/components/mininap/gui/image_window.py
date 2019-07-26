from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QAction, qApp, QMenuBar

from .image_widget import ImageWidget
from ..image.napari_image import NImage


class ImageWindow(QMainWindow):
    def __init__(self, image: NImage, window_width=800, window_height=800, parent=None):

        super(ImageWindow, self).__init__(parent)

        self.widget = ImageWidget(
            image, window_width, window_height, containing_window=self
        )

        self.setCentralWidget(self.widget)

        self.show()
        self.raise_()

        self.installEventFilter(self)

    # def eventFilter(self, object, event):
    #     #print(event)
    #     #print(event.type())
    #     if event.type() == QEvent.ShortcutOverride:
    #         #print(event)
    #         #print(event.key())
    #         if (event.key() == Qt.Key_F or event.key() == Qt.Key_Enter) and not self.isFullScreen():
    #             print("showFullScreen!")
    #             self.showFullScreen()
    #         elif (event.key() == Qt.Key_F or event.key() == Qt.Key_Escape) and self.isFullScreen():
    #             print("showNormal!")
    #             self.showNormal()
    #
    #     return QWidget.eventFilter(self, object, event)

    def update_image(self):
        self.widget.update_image()

    def set_cmap(self, cmap):
        self.widget.set_cmap(cmap)
