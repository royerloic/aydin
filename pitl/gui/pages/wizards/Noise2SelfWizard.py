from PyQt5.QtWidgets import QTabWidget

from pitl.gui.components.tabs.roi import ROIPage
from pitl.gui.components.tabs.run_n2s import RunN2STab
from pitl.gui.components.tabs.test_n2s import TestN2STab
from pitl.gui.components.tabs.upload_n2s import UpdateN2STab


class Noise2SelfWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        # Add tabs
        self.upload_tab = UpdateN2STab(self)
        self.addTab(self.upload_tab, "Upload")

        # self.roi_image = self.upload_tab.input_lbl
        self.roi_tab = ROIPage(self, QTabWidget())
        self.addTab(self.roi_tab, "ROI")

        self.test_tab = TestN2STab(self, f)
        self.addTab(self.test_tab, "Test")

        self.run_tab = RunN2STab(self, f)
        self.addTab(self.run_tab, "Run")

    def next_tab(self):
        self.setCurrentIndex(self.currentIndex() + 1)

    def prev_tab(self):
        self.setCurrentIndex(self.currentIndex() - 1)
