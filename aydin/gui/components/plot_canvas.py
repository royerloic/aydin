from PyQt5.QtWidgets import QWidget, QGridLayout
import pyqtgraph as pg


class PlotCanvas(QWidget):
    def __init__(self, parent):
        super(PlotCanvas, self).__init__(parent)

        self.layout = QGridLayout()

        self.pw = pg.PlotWidget(name='Plot1')
        self.layout.addWidget(self.pw)
        self.setLayout(self.layout)

        self.values = []
        self.pw.plot(self.values, pen=None, symbol='o')

    def add_val(self, val):
        self.values += [val]
        self.pw.plot(self.values, clear=True)
