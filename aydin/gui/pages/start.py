from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget,
    QSizePolicy,
    QLabel,
    QFrame,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
)

from aydin.gui.pages.wizards.Noise2SelfWizard import Noise2SelfWizard
from aydin.gui.pages.wizards.Noise2TruthWizard import Noise2TruthWizard


class StartPage(QWidget):
    def __init__(self, parent, threadpool):
        super(QWidget, self).__init__(parent)
        self.parent = parent
        self.threadpool = threadpool

        # Layout
        self.layout = QVBoxLayout()

        # Title
        self.title_label = QLabel("AYDIN")
        self.title_label.setFont(QFont("Arial", 146, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Horizontal Line Break
        self.hline_break = QFrame()
        self.hline_break.setFrameShape(QFrame.HLine)
        self.hline_break.setFrameShadow(QFrame.Sunken)
        self.layout.addWidget(self.hline_break)

        # Buttons for different services
        self.buttons_layout = QHBoxLayout()
        self.n2s_button = QPushButton("Denoise \n(Noise2Self)")
        self.n2s_button.setFont(QFont("Arial", 26, QFont.Bold))
        self.n2s_button.pressed.connect(self.switch_to_n2s)
        self.n2s_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.buttons_layout.addWidget(self.n2s_button)

        # self.n2t_button = QPushButton("Translate \n(Noise2Truth)")
        # self.n2t_button.setFont(QFont("Arial", 26, QFont.Bold))
        # self.n2t_button.pressed.connect(self.switch_to_n2t)
        # self.n2t_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # self.buttons_layout.addWidget(self.n2t_button)

        self.layout.addLayout(self.buttons_layout)

        # Set final layout as widget layout
        self.setLayout(self.layout)

    def switch_to_n2s(self):
        self.parent.setCentralWidget(Noise2SelfWizard(self, self.threadpool))

    def switch_to_n2t(self):
        self.parent.setCentralWidget(Noise2TruthWizard(self, self.threadpool))
