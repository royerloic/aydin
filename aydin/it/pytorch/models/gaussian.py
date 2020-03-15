import numpy
import scipy
import torch
from torch import nn


class GaussianLayer(nn.Module):
    def __init__(
        self,
        num_channels_in=1,
        num_channels_out=1,
        kernel_size=21,
        sigma=3,
        donut=False,
    ):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d((kernel_size - 1) // 2),
            nn.Conv2d(
                num_channels_in,
                num_channels_out,
                kernel_size,
                stride=1,
                padding=0,
                bias=False,
                groups=num_channels_in,
            ),
        )

        self.weights_init(kernel_size, sigma, donut)

    def weights_init(self, kernel_size, sigma, donut):
        kernel = numpy.zeros((kernel_size, kernel_size))
        center = (kernel_size - 1) // 2
        kernel[center, center] = 1
        kernel = scipy.ndimage.gaussian_filter(kernel, sigma=sigma)
        if donut:
            kernel[center, center] = 0

        # import napari
        # with napari.gui_qt():
        #     viewer = napari.Viewer()
        #     viewer.add_image(kernel, name='kernel')

        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))

    def forward(self, x):
        return self.seq(x)
