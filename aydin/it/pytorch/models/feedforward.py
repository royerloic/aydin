import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self, n_input_channel=1, n_output_channel=1, depth=8, nic=8, kernel_size=3
    ):
        super().__init__()

        self.convs = []
        self.norms = []
        self.nonlins = []
        # nn.Dropout2d(

        for i in range(0, depth - 1):
            in_channels = n_input_channel if i == 0 else nic
            conv = nn.Conv2d(
                in_channels,
                nic,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
            self.convs.append(conv)

            norm = nn.InstanceNorm2d(nic)
            self.norms.append(norm)

            relu = nn.Hardtanh(inplace=True)
            self.nonlins.append(relu)

            #

        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.nonlins = nn.ModuleList(self.nonlins)
        self.final_conv = nn.Conv2d(nic, n_output_channel, kernel_size=1, padding=0)

    def forward(self, x0):

        x = x0

        xn = []
        for conv, norm, nonlin in zip(self.convs, self.norms, self.nonlins):
            x = nonlin(norm(conv(x)))
            xn.append(x)

        y = xn[0]
        for x in xn[1:]:
            y = y + x

        y = self.final_conv(y)

        return y
