import os
import torch
from torch import nn
import pandas as pd
import numpy as np


class EEGBCI_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(EEGBCI_Dataset, self).__init__()

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
        data = self.data[item].reshape(-1, 10)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class EEGBCI_FC(nn.Module):
    def __init__(self, dropout_rate):
        super(EEGBCI_FC, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.reshape(-1, 10)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out