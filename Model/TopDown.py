import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.reduce_dim_5 = nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=1, padding=0)
        self.reduce_dim_4 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=1, padding=0)
        self.reduce_dim_3 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=1, padding=0)
        self.reduce_dim_2 = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=1, padding=0)

        self.double_conv_5 = self._make_dobule_conv(256, 128)
        self.double_conv_4 = self._make_dobule_conv(256, 128)
        self.double_conv_3 = self._make_dobule_conv(256, 128)
        self.double_conv_2 = self._make_dobule_conv(256, 128)

    def _up_add(self, x, y):
        (_, _, H, W) = y.size()
        x_up = nn.functional.upsample(x, size=(H, W), mode='nearest')

        return x_up + y

    def _make_dobule_conv(self, in_dims, out_dims):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),
            nn.Conv2d(out_dims, out_dims, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_dims),
            nn.ReLU()
        )

        return conv_layer

    def forward(self, c2, c3, c4, c5):
        m5 = self.reduce_dim_5(c5)
        m4 = self._up_add(m5, self.reduce_dim_4(c4))
        m3 = self._up_add(m4, self.reduce_dim_3(c3))
        m2 = self._up_add(m3, self.reduce_dim_2(c2))

        m5 = self.double_conv_5(m5)
        m4 = self.double_conv_4(m4)
        m3 = self.double_conv_3(m3)
        m2 = self.double_conv_2(m2)

        return m5, m4, m3, m2