import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, name='resnet50'):
        super(Encoder, self).__init__()

        model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=True)
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

        m2 = self.res_block_1(x)
        m3 = self.res_block_2(m2)
        m4 = self.res_block_3(m3)
        m5 = self.res_block_4(m4)

        return [m2, m3, m4, m5]


a = Encoder()