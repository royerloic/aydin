import numpy as np
import sys

from PyQt5.QtWidgets import QWidget
from vispy import gloo, app, visuals, scene

CustomLineVisual = scene.visuals.create_visual_node(visuals.LinePlotVisual)


class PlotCanvas(QWidget):
    def __init__(self, parent):
        # build canvas
        self.canvas = scene.SceneCanvas(keys=None, show=True)

        # Add a ViewBox to let the user zoom/rotate
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.interactive = False
        self.view.camera.distance = 60

        N = 10
        pos = np.zeros((N, 2), dtype=np.float32)
        self.x = np.linspace(0.10, 0.2, N)
        self.y = np.random.normal(size=N, scale=0.1, loc=0.5)

        # plot
        self.pos = np.c_[self.x, self.y]
        self.line = CustomLineVisual(
            self.pos,
            width=2.0,
            color='red',
            edge_color='w',
            symbol='o',
            face_color=(0.2, 0.2, 1, 0.8),
            parent=self.view.scene,
        )

    def add_pos(self, N):
        self.pos = np.zeros((N // 10, 2), dtype=np.float32)
        self.x = np.linspace(0.1, (N / 500) * 1.5, N // 10)
        self.y = np.random.normal(size=N // 10, scale=0.1, loc=0.5)

        # plot
        self.pos = np.c_[self.x, self.y]
        self.line.set_data(self.pos)


if __name__ == '__main__':
    win = PlotCanvas()

    if sys.flags.interactive != 1:
        app.run()
