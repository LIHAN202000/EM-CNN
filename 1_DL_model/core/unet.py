import torch
import torch.nn as nn

from core.modules import ResidualConv, Upsample


class Unet(nn.Module):
    def __init__(self, channel, filters=None):
        super().__init__()

        if filters is None:
            filters = [64, 128, 256, 512, 1024]

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=(3, 3), stride=(1, 1), padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], (2, 2), 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], (2, 2), 1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], (2, 2), 1)

        self.bridge = ResidualConv(filters[3], filters[4], (2, 2), 1)

        self.upsample_0 = Upsample(filters[4], filters[4], (2, 2), (2, 2))
        self.up_residual_conv0 = ResidualConv(filters[4] + filters[3], filters[3], (1, 1), 1)

        self.upsample_1 = Upsample(filters[3], filters[3], (2, 2), (2, 2))
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], (1, 1), 1)

        self.upsample_2 = Upsample(filters[2], filters[2], (2, 2), (2, 2))
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], (1, 1), 1)

        self.upsample_3 = Upsample(filters[1], filters[1], (2, 2), (2, 2))
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], (1, 1), 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], channel, (1, 1), (1, 1)),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x3_ = self.residual_conv_3(x3)
        # Bridge
        x4_ = self.bridge(x3_)
        # Decode
        x4_ = self.upsample_0(x4_)
        x4 = torch.cat([x4_, x3_], dim=1)

        x4 = self.up_residual_conv0(x4)

        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output


