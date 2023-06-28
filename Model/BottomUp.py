import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, backbone='resnet152'):
        super(Encoder, self).__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)
        self._first_conv = self._make_first_conv(model)

        self.res_block_1 = model.layer1
        self.res_block_2 = model.layer2
        self.res_block_3 = model.layer3
        self.res_block_4 = model.layer4

    def _make_first_conv(self, model):
        fist_conv_module = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        return fist_conv_module

    def forward(self, x):
        x = self._first_conv(x)

        c2 = self.res_block_1(x)
        c3 = self.res_block_2(c2)
        c4 = self.res_block_3(c3)
        c5 = self.res_block_4(c4)

        return c2, c3, c4, c5
