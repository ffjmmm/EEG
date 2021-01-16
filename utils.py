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
        self.label = self.data[:, -2:]
        self.data = self.data[:, :-2]
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item].reshape(4, 800)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class EEGBCI_CNN(nn.Module):
    def __init__(self, dropout_rate):
        super(EEGBCI_CNN, self).__init__()

        # input 1 x 4 x 800
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )

        # 64 * 2 * 400
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )

        # 128 * 1 * 200
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 200, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 128 * 200)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def preprocess(data, method):
    num_channel = data.shape[1]
    len_data = data.shape[2]
    if method == 'None':
        return data.reshape(-1, num_channel * len_data)

    for i in range(len(data)):
        for j in range(num_channel):
            mean = data[i][j].mean()
            std = data[i][j].std()
            data_max = data[i][j].max()
            data_min = data[i][j].min()
            for k in range(len_data):
                if method == 'Normalization':
                    data[i][j][k] = (data[i][j][k] - mean) / std
                else:
                    data[i][j][k] = (data[i][j][k] - data_min) / (data_max - data_min)
    return data.reshape(-1, num_channel * len_data)