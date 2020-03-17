from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipNet2D(nn.Module):
    def __init__(
        self,
        num_input_channels=1,
        num_output_channels=1,
        kernel_sizes=[7, 5, 3, 3, 3, 3, 3, 3],
        num_features=[64, 32, 16, 8, 4, 2, 1, 1],
        layers=6,
        num_channels=None,
    ):
        super().__init__()

        self.training = True
        assert len(kernel_sizes) == len(num_features)
        self.kernel_sizes = kernel_sizes
        self.num_features = num_features
        self.num_scales = len(kernel_sizes)
        self.skipconvlist = []

        total_num_features = 0
        current_radius = 0
        current_num_channels = num_input_channels
        for scale_index in range(self.num_scales):
            print(f"current_radius={current_radius}")
            size = kernel_sizes[scale_index]
            num = num_features[scale_index]

            radius = (size - 1) // 2
            dilation = 1 + current_radius

            skipconv = nn.Conv2d(
                current_num_channels,
                num,
                kernel_size=size,
                padding=dilation * radius,
                dilation=dilation,
                padding_mode='zeros',
                groups=min(current_num_channels, num),
            )

            # self.initialise_skipconv(size, skipconv)

            total_num_features += num
            current_radius += dilation * radius
            current_num_channels = num
            print(skipconv)
            self.skipconvlist.append(skipconv)

        self.total_num_features = total_num_features
        print(f"total_num_features={total_num_features}")

        self.skipconvlist = nn.ModuleList(self.skipconvlist)

        if num_channels is None:
            num_channels = total_num_features

        self.denseconvlist = nn.ModuleList()

        for level in range(0, layers):
            print("Level: %d" % level)
            num_in = total_num_features if level == 0 else num_channels
            num_out = num_output_channels if level == layers - 1 else num_channels
            dense = nn.Conv2d(num_in, num_out, kernel_size=1, padding=0)
            print(dense)
            self.denseconvlist.append(dense)

        self.finalconv = nn.Conv2d(
            num_channels, num_output_channels, kernel_size=1, padding=0
        )

    def initialise_skipconv(self, size, skipconv):
        with torch.no_grad():
            skipconv_shape = skipconv._parameters['weight'].shape
            nb_filters = skipconv_shape[0]
            nb_channels = skipconv_shape[1]

            skipconv._parameters['bias'] *= 0
            # skipconv._parameters['weight'][:, :, :, :] = 0

            if nb_filters < size * size:
                skipconv._parameters['weight'][:, :, :, :] = 1 / (size * size)
            else:
                index = 0
                for i in range(0, size):
                    for j in range(0, size):
                        if i != j:
                            index += 1
                            if index >= nb_filters:
                                break
                            skipconv._parameters['weight'][
                                index, :, i, j
                            ] = 1 + 0.001 * (random() - 0.5)

    def forward(self, x0):

        x = x0
        features = []
        for scale_index in range(self.num_scales):
            skipconv = self.skipconvlist[scale_index]
            x = skipconv(x)
            features.append(x)
            x = F.leaky_relu(x, inplace=True, negative_slope=0.01)

        x = torch.cat(features, 1)

        y = None
        for dense in self.denseconvlist:
            x = dense(x)
            x = F.leaky_relu(x, inplace=True, negative_slope=0.01)
            if y is None:
                y = x
            else:
                y = y + x

        y = self.finalconv(y)

        return y

    def trainable_parameters(self):
        from itertools import chain

        return chain(
            self.skipconvlist.parameters(),
            self.denseconvlist.parameters(),
            self.finalconv.parameters(),
        )

    def post_optimisation(self):
        with torch.no_grad():
            for skipconv in self.skipconvlist[0:1]:  #
                indexes = tuple((i - 1) // 2 for i in skipconv.kernel_size)
                skipconv._parameters['weight'][:, :, indexes[0], indexes[1]] = 0
