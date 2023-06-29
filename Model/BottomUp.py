import torch
import torch.nn as nn
import torchsummary


class Encoder(nn.Module):
    def __init__(self, backbone='resnet152'):
        super(Encoder, self).__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)

        self._first_conv = self._make_first_conv()

        self.res_block_1 = self.model.layer1
        self.res_block_2 = self.model.layer2
        self.res_block_3 = self.model.layer3
        self.res_block_4 = self.model.layer4

    def _make_first_conv(self):
        module = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool
        )

        return module

    def forward(self, x):
        x = self._first_conv(x)
        c2 = self.res_block_1(x)
        c3 = self.res_block_2(c2)
        c4 = self.res_block_3(c3)
        c5 = self.res_block_4(c4)

        return c2, c3, c4, c5
