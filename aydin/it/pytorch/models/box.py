import numpy
import scipy
import torch
from torch import nn


class BoxBlurLayer(nn.Module):
    def __init__(
        self,
        num_channels_in=1,
        num_channels_out=1,
        kernel_size=21,
        donut=False,
        separate=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            num_channels_in,
            num_channels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
            groups=num_channels_in if separate else 1,
        )

        self.weights_init(kernel_size, donut)

    def weights_init(self, kernel_size, donut):
        kernel = numpy.ones((kernel_size, kernel_size))
        if kernel_size > 1:
            center = (kernel_size - 1) // 2
            if donut:
                kernel[center, center] = 0
            kernel /= kernel_size ** 2 + (-1 if donut else 0)

        # import napari
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_image(kernel, name='kernel')

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))

    def forward(self, x):
        return self.conv(x)
