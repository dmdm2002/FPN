import torch
import torch.nn as nn

from Model.BottomUp import Encoder
from Model.TopDown import Decoder

import torchsummary


class FPN(nn.Module):
    def __init__(self, backbone='resnet152'):
        super(FPN, self).__init__()
        self.encoder = Encoder(backbone=backbone)
        self.decoder = Decoder()

        self.conv_5 = self._conv_layer(256, 128)
        self.conv_4 = self._conv_layer(256, 128)
        self.conv_3 = self._conv_layer(256, 128)
        self.conv_2 = self._conv_layer(256, 128)

        self.output = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)

    def _up_size(self, x, size):
        (H, W) = size
        return nn.functional.upsample(x, size=(H, W), mode='nearest')

    def _conv_layer(self, in_dims, out_dims):
        module = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),
        )

        return module

    def forward(self, x):
        (_, _, H, W) = x.shape
        size = (H, W)
        c2, c3, c4, c5 = self.encoder(x)
        p5, p4, p3, p2 = self.decoder(c2, c3, c4, c5)

        p5 = self._up_size(self.conv_5(p5), size)
        p4 = self._up_size(self.conv_4(p4), size)
        p3 = self._up_size(self.conv_3(p3), size)
        p2 = self._up_size(self.conv_2(p2), size)

        concat_p = torch.cat([p5, p4, p3, p2], dim=1)

        output = self.output(concat_p)

        return output

model = FPN()
torchsummary.summary(model, (3, 224, 224), device='cpu')