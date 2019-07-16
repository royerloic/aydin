import numpy as np
import sys

from vispy import gloo, app, visuals


class PlotCanvas(app.Canvas):
    def __init__(self, parent):
        # vertex positions of data to draw
        app.Canvas.__init__(self, keys=None, vsync=True, size=(600, 600))
        super().__init__(parent)

        self.line = None

    def on_draw(self, event):
        gloo.clear('black')
        if self.line is not None:
            self.line.draw()

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        if self.line is not None:
            self.line.transforms.configure(canvas=self, viewport=vp)

    def add_pos(self, N):
        self.pos = np.zeros((N, 2), dtype=np.float32)
        self.pos[:, 0] = np.linspace(10, 590, N)
        self.pos[:, 1] = np.random.normal(size=N, scale=50, loc=150)
        print(self.pos)
        self.line = visuals.LinePlotVisual(
            self.pos,
            color='w',
            edge_color='w',
            symbol='o',
            face_color=(0.2, 0.2, 1),
            marker_size=5,
        )


if __name__ == '__main__':
    win = PlotCanvas()

    if sys.flags.interactive != 1:
        app.run()
