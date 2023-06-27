import torch
import torch.nn as nn
import torchvision


class Encoder(nn.Module):
    def __init__(self, name='resnet50'):
        super(Encoder, self).__init__()
        # model = torch.hub.load('pytorch/vision:v0.10.0', name, pretrained=True)
        model = torchvision.models.resnet152(pretrained=True)
        # layer_1 = model[:]
        # print(model.features[:5])
        print(model)

    def forward(self, x):
        pass


a = Encoder()