import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from aydin.it.pytorch.models.gaussian import GaussianLayer


class SlimNet(nn.Module):
    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
        kernel_size=9,
        levels=4,
        norm='batch',
    ):
        super(SlimNet, self).__init__()

        self.kernel_size = kernel_size

        num_channels = int(kernel_size * kernel_size)

        padding = int((kernel_size - 1) / 2)

        print("kernel_size= %d" % kernel_size)
        print("num_channels= %d" % num_channels)

        self.gaussian = GaussianLayer(kernel_size=11, sigma=1.5)

        self.convplist = nn.ModuleList()
        self.normlist = nn.ModuleList()

        for level in range(0, levels):
            print("Level: %d" % level)
            num_in = num_input_channels if level == 0 else num_channels
            num_out = num_output_channels if level == levels - 1 else num_channels
            convp = nn.Conv2d(num_in, num_out, kernel_size=1, padding=0)
            print(convp)
            self.convplist.append(convp)

        print(self.convfinal)

    def set_weights(self, weights, bias=0):
        device = next(self.parameters()).device
        with torch.no_grad():
            length = weights.shape[0]
            print(length)
            self.convspatial._parameters['weight'][0:length] = torch.from_numpy(
                weights
            ).to(device)
            # self.convspatial._parameters['bias']   = bias*torch.ones([self.num_channels], dtype=torch.float32, device=device)

    def set_donut(self):
        with torch.no_grad():
            n = self.kernel_size
            index = (n - 1) // 2

            self.convspatial._parameters['weight'][:, :, index, index] = 0

    def set_trivial_basis(self):
        with torch.no_grad():
            device = next(self.parameters()).device
            n = self.kernel_size

            self.convspatial._parameters['bias'] *= 0

            self.convspatial._parameters['weight'][:, :, :, :] = 0
            for i in range(0, n):
                for j in range(0, n):
                    index = i * n + j
                    self.convspatial._parameters['weight'][index, 0, i, j] = 1

    def trainable_parameters(self):
        from itertools import chain

        parameters_list = (
            [convp.parameters() for convp in self.convplist]
            + [self.convfinal.parameters()]
            + [norm.parameters() for norm in self.normlist]
        )
        print(parameters_list)
        return chain(*parameters_list)

    def forward(self, x):

        x = self.convspatial(x)
        x = self.convrndf(x)
        x = F.leaky_relu(x)

        for (convp, norm) in zip(self.convplist, self.normlist):
            x = convp(x)
            if norm:
                x = norm(x)
            x = F.leaky_relu(x)

        prediction = self.convfinal(x)

        return prediction
