from PyQt5.QtWidgets import QTabWidget

from aydin.gui.components.tabs.load.load_n2t import LoadN2TTab
from aydin.gui.components.tabs.roi.roi import ROITab
from aydin.gui.components.tabs.train.train_n2t import TrainN2TTab


class Noise2TruthWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)
        self.parent = parent

        self.tabBar().hide()

        self.currentChanged.connect(self.handle_tab_change)
        self.monitor_images = (
            []
        )  # TODO: check the best scheme for handling this in the end

        # Add tabs
        self.upload_tab = LoadN2TTab(self)
        self.addTab(self.upload_tab, "Load")

        self.roi_tab = ROITab(self)
        self.addTab(self.roi_tab, "Region-Of-Interest")

        self.test_tab = TrainN2TTab(self, f)
        self.addTab(self.test_tab, "Train")

        # self.run_tab = PredictN2TTab(self, f)
        # self.addTab(self.run_tab, "Predict")

        # Disable all tabs except first one
        self.setTabEnabled(1, False)
        self.setTabEnabled(2, False)
        self.setTabEnabled(3, False)
        self.setTabEnabled(4, False)
        self.setTabEnabled(5, False)

    def next_tab(self):
        self.setCurrentIndex(self.currentIndex() + 1)

    def prev_tab(self):
        self.setCurrentIndex(self.currentIndex() - 1)

    def handle_tab_change(self):
        idx = self.currentIndex()
        self.setTabEnabled(idx, True)

        if idx == 1:
            if not self.roi_tab.is_loaded:
                self.roi_tab.load_tab()
        elif idx == 2:
            if not self.test_tab.is_loaded:
                self.test_tab.load_tab()
