import math
from collections import Iterator

import numpy
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class LucyRichardson(nn.Module):
    def __init__(
        self, kernel_psf, num_channels_in=1, num_channels_out=1, iterations=4, clip=True
    ):
        super().__init__()

        self.clip = clip
        self.num_channels_out = num_channels_out
        self.num_channels_in = num_channels_in
        self.iterations = iterations

        self.psf_size = kernel_psf.shape[0]

        kernel_psf = kernel_psf.astype(numpy.float)
        kernel_psf_mirror = kernel_psf[::-1, ::-1].copy()

        self.kernel_psf_tensor = torch.from_numpy(
            kernel_psf[numpy.newaxis, numpy.newaxis, ...]
        ).float()
        self.kernel_psf_mirror_tensor = torch.from_numpy(
            kernel_psf_mirror[numpy.newaxis, numpy.newaxis, ...]
        ).float()

        self.kernel_psf_tensor = torch.nn.Parameter(self.kernel_psf_tensor)
        self.kernel_psf_mirror_tensor = torch.nn.Parameter(
            self.kernel_psf_mirror_tensor
        )

    def forward(self, x):

        im_deconv = 0.5 * torch.ones_like(x)

        for _ in range(self.iterations):
            convolved = F.conv2d(
                im_deconv, self.kernel_psf_tensor, padding=(self.psf_size - 1) // 2
            )
            relative_blur = x / convolved
            im_deconv = im_deconv * F.conv2d(
                relative_blur,
                self.kernel_psf_mirror_tensor,
                padding=(self.psf_size - 1) // 2,
            )

        if self.clip:
            im_deconv.clamp_(-1, 1)

        return im_deconv

    def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
        return chain

    def post_optimisation(self):
        with torch.no_grad():
            self.kernel_psf_tensor += torch.min(self.kernel_psf_tensor)
            self.kernel_psf_tensor /= torch.sum(self.kernel_psf_tensor)
