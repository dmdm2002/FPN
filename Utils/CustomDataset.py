import glob
import numpy as np

import PIL.Image as Image
from torch.utils import data


class CustomDB(data.Dataset):
    def __init__(self, dataset_path, run_type, transform):
        super(CustomDB, self).__init__()

        self.transform = transform

        images = glob.glob(f'{dataset_path}/{run_type}/images/*.jpg')
        labels = glob.glob(f'{dataset_path}/{run_type}/labels/*.jpg')

        self.path = list(zip(images, labels))

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.path[idx][0]))
        label = np.array(Image.open(self.path[idx][1]))[0] / 255

        img = self.transform(img)
        label = self.transform(label)

        return [img, label]

