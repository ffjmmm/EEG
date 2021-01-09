import os
import torch
import pandas as pd
import numpy as np


class EEGBCIDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(EEGBCIDataset, self).__init__()

        if train:
            self.data = pd.read_csv(os.path.join(root, 'train_data.csv'))
        else:
            self.data = pd.read_csv(os.path.join(root, 'test_data.csv'))
        self.num = len(self.data)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = self.data[:, -1:]
        self.data = self.data[:, :-1]
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item]



data = pd.read_csv('./test_data.csv')
print(len(data))
data = np.asarray(data, dtype=np.float)
label = data[:, -1:]
data = data[:, :-1]
print(data.shape)
print(label)