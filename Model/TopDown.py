import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reduce_dim_1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.reduce_dim_2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.reduce_dim_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.reduce_dim_4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.out_layer_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.out_layer_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.out_layer_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _up_add(self, x, y):
        (_, _, H, W) = y.size()
        x_up = nn.functional.upsample(x, size=(H, W), mode='nearest')

        return x_up + y

    def forward(self, x):
        (c2, c3, c4, c5) = x

        p5 = self.reduce_dim_4(c5)
        m4 = self._up_add(p5, self.reduce_dim_3(c4))
        m3 = self._up_add(m4, self.reduce_dim_3(c3))
        m2 = self._up_add(m3, self.reduce_dim_3(c2))

        # output layer
        p4 = self.out_layer_3(m4)
        p3 = self.out_layer_2(m3)
        p2 = self.out_layer_1(m2)

        return [p5, p4, p3, p2]
