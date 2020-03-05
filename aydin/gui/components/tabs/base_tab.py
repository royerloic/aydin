from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame


class BaseTab(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.wizard = parent
        self.base_layout = QVBoxLayout()
        self.base_layout.addStretch(1)

        # Horizontal Line Break
        self.wizard_ops_layout = QVBoxLayout()
        self.hline_break = QFrame()
        self.hline_break.setFrameShape(QFrame.HLine)
        self.hline_break.setFrameShadow(QFrame.Sunken)
        self.wizard_ops_layout.addWidget(self.hline_break)

        # Finalize base layout
        self.base_layout.addLayout(self.wizard_ops_layout)
        self.setLayout(self.base_layout)

    def load_tab(self):
        pass
