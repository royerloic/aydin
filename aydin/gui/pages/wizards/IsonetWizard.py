from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QTabWidget

from aydin.gui.components.tabs.test_n2s import TestN2STab
from aydin.gui.components.tabs.test_n2t import TestN2TTab


class IsonetWizard(QTabWidget):
    def __init__(self, parent, f):
        super(QTabWidget, self).__init__(parent)

        # Add tabs
        self.upload_tab = TestN2STab(self, f)
        self.addTab(self.upload_tab, "Upload")

        self.roi_tab = TestN2TTab(self)
        self.addTab(self.roi_tab, "ROI")

        self.test_tab = TestN2STab(self, f)
        self.addTab(self.test_tab, "Test")

        self.run_tab = TestN2STab(self, f)
        self.addTab(self.run_tab, "Run")

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(
                currentQTableWidgetItem.row(),
                currentQTableWidgetItem.column(),
                currentQTableWidgetItem.text(),
            )
