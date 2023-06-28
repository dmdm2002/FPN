import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from Model.FPN import FPN
from Utils.Managers import CkpManager, TransformManager
from Utils.Displayer import Displayer
from Utils.Options import Param
from Utils.CustomDataset import CustomDB

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Trainer(Param):
    def __init__(self):
        super(Trainer, self).__init__()
        os.makedirs(self.output_log, exist_ok=True)
        self.ckp_manager = CkpManager()
        self.transformManager = TransformManager()

    def run(self):
        print('----------------------------------------------------------')
        print(f'[Device]: {self.device}')
        print('----------------------------------------------------------')

        model = FPN(backbone=self.backbone)
        model, epoch = self.ckp_manager.load_ckp(model)

        model = model.to(self.device)
        model.train()

        transform = self.transformManager.set_transform(to_image=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            list(model.parameters()),
            lr=self.lr,
        )

        tr_dataset = CustomDB(self.db_path, run_type=self.run_type[0], transform=transform)
        te_dataset = CustomDB(self.db_path, run_type=self.run_type[0], transform=transform)

        tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.batchsz, shuffle=True)
        te_loader = DataLoader(dataset=te_dataset, batch_size=self.batchsz, shuffle=False)

