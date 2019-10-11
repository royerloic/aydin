from PyQt5.QtWidgets import QTabWidget

from aydin.gui.components.tabs.roi.roi import ROITab
from aydin.gui.components.tabs.predict.predict_n2s import PredictN2STab
from aydin.gui.components.tabs.train.train_n2s import TrainN2STab
from aydin.gui.components.tabs.load.load_n2s import LoadN2STab


class Noise2SelfWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        self.currentChanged.connect(self.handle_tab_change)
        self.monitor_images = (
            []
        )  # TODO: check the best scheme for handling this in the end

        # Add tabs
        self.upload_noisy_tab = LoadN2STab(self)
        self.addTab(self.upload_noisy_tab, "Load")

        self.roi_tab = ROITab(self)
        self.addTab(self.roi_tab, "Region-Of-Interest")

        self.test_tab = TrainN2STab(self, f)
        self.addTab(self.test_tab, "Train")

        # self.run_tab = PredictN2STab(self, f)
        # self.addTab(self.run_tab, "Predict")

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
        self.setTabEnabled(idx, True)

        if idx == 1:
            if not self.roi_tab.is_loaded:
                self.roi_tab.load_tab()
        elif idx == 2:
            if not self.test_tab.is_loaded:
                self.test_tab.load_tab()
