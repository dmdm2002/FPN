import torch
import torch.nn as nn

from Model2.BottomUp import Encoder
from Model2.TopDown import Decoder

import torchsummary


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, n_classes=2):
        super(FeaturePyramidNetwork, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.output = nn.Conv2d(512, n_classes, kernel_size=(3, 3), stride=1, padding=1)

    def _up_size(self, x):
        return nn.functional.upsample(x, size=(224, 224), mode='nearest')

    def forward(self, x):
        c2, c3, c4, c5 = self.encoder(x)
        m5, m4, m3, m2 = self.decoder(c2, c3, c4, c5)
        p5 = self._up_size(m5)
        p4 = self._up_size(m4)
        p3 = self._up_size(m3)
        p2 = self._up_size(m2)

        p_cat = torch.cat([p5, p4, p3, p2], dim=1)
        out = self.output(p_cat)

        return out

model = FeaturePyramidNetwork()
torchsummary.summary(model, (3, 224, 224), device='cpu')