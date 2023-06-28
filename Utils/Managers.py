import os
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Utils.Options import Param


class CkpManager(Param):
    def __init__(self):
        super(CkpManager, self).__init__()

    def init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant(module.bias.data, 0.0)

    def load_ckp(self, model):
        if self.do_load_ckp:
            print(f'Check Point [self.load_epoch] Loading...')

            ckp = torch.load(f'{self.output_ckp}/{self.load_epoch}.pth')
            time.sleep(0.1)

            model.load_state_dict(ckp['model_state_dict'])
            epoch = ckp['epoch']+1

        else:
            print("Initialize Model Weight...")
            model.apply(self.init_weight)
            epoch = 0

        return model, epoch

    def save_ckp(self, model, epoch):
        os.makedirs(self.output_ckp, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch
            },
            f'{self.output_ckp}/{epoch}.pth'
        )


class TransformManager(Param):
    def __init__(self):
        super(TransformManager, self).__init__()

    def set_transform(self, to_image=False):
        assert type(to_image) is bool, 'Only boolean type is available for to_image.'

        if to_image:
            transform = transforms.Compose([
                transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                transforms.ToPILImage(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        return transform