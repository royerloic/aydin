from PyQt5.QtWidgets import QTabWidget

from pitl.gui.components.tabs.test_n2t import TestN2TTab


class Noise2TruthWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        # Add tabs
        self.upload_tab = TestN2TTab(self, f)
        self.addTab(self.upload_tab, "Upload")

        self.roi_tab = TestN2TTab(self)
        self.addTab(self.roi_tab, "ROI")

        self.test_tab = TestN2TTab(self, f)
        self.addTab(self.test_tab, "Test")

        self.run_tab = TestN2TTab(self, f)
        self.addTab(self.run_tab, "Run")
