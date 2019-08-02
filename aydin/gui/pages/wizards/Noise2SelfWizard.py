import numpy as np
from PyQt5.QtWidgets import QTabWidget

from aydin.gui.components.tabs.roi import ROIPage
from aydin.gui.components.tabs.run_n2s import RunN2STab
from aydin.gui.components.tabs.test_n2s import TestN2STab
from aydin.gui.components.tabs.upload_n2s import UploadN2STab


class Noise2SelfWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        self.currentChanged.connect(self.handle_tab_change)
        self.monitor_image = (
            None
        )  # TODO: check the best scheme for handling this in the end

        # Add tabs
        self.upload_tab = UploadN2STab(self)
        self.addTab(self.upload_tab, "Upload")

        self.roi_tab = ROIPage(self)
        self.addTab(self.roi_tab, "ROI")

        self.test_tab = TestN2STab(self, f)
        self.addTab(self.test_tab, "Test")

        self.run_tab = RunN2STab(self, f)
        self.addTab(self.run_tab, "Run")

        # Disable all tabs except first one
        self.setTabEnabled(1, False)
        self.setTabEnabled(2, False)
        self.setTabEnabled(3, False)

    def next_tab(self):
        self.setCurrentIndex(self.currentIndex() + 1)

    def prev_tab(self):
        self.setCurrentIndex(self.currentIndex() - 1)

    def handle_tab_change(self):
        idx = self.currentIndex()
        if idx == 0:
            self.setTabEnabled(0, True)
            self.setTabEnabled(1, False)
            self.setTabEnabled(2, False)
            self.setTabEnabled(3, False)
        elif idx == 1:
            self.roi_tab.load_tab()
            self.setTabEnabled(1, True)
            self.setTabEnabled(2, False)
            self.setTabEnabled(3, False)
        elif idx == 2:
            self.test_tab.load_tab()
            self.setTabEnabled(2, True)
            self.setTabEnabled(3, False)
        elif idx == 3:
            self.setTabEnabled(3, True)
