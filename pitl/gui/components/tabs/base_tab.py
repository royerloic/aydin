from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QPushButton,
)


class BaseTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.wizard = parent
        self.base_layout = QVBoxLayout()

        # Horizontal Line Break
        self.wizard_ops_layout = QVBoxLayout()
        self.hline_break = QFrame()
        self.hline_break.setFrameShape(QFrame.HLine)
        self.hline_break.setFrameShadow(QFrame.Sunken)
        self.wizard_ops_layout.addWidget(self.hline_break)

        # Wizard navigation buttons
        self.buttons_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.pressed.connect(self.wizard.prev_tab)
        self.next_button.pressed.connect(self.wizard.next_tab)
        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.next_button)
        self.wizard_ops_layout.addLayout(self.buttons_layout)

        # Finalize base layout
        self.base_layout.addLayout(self.wizard_ops_layout)
        self.setLayout(self.base_layout)

    def load_tab(self):
        pass
