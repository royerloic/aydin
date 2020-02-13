import torch.nn as nn


class SingleConvolution(nn.Module):
    def __init__(self, n_channel_in=1, n_channel_out=1, width=3):
        super(SingleConvolution, self).__init__()

        self.conv = nn.Conv2d(
            n_channel_in, n_channel_out, kernel_size=width, padding=width // 2
        )

    def forward(self, x):
        return self.conv(x)
