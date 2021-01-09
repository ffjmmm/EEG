import pandas as pd
import torch
from torch import nn
import os
import numpy as np


class PhysioNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(PhysioNetDataset, self).__init__()

        if train:
            self.data = pd.read_csv(os.path.join(root, 'Training_data.csv'), header=None)
            self.label = pd.read_csv(os.path.join(root, 'Training_labels.csv'), header=None)
        else:
            self.data = pd.read_csv(os.path.join(root, 'Test_data.csv'), header=None)
            self.label = pd.read_csv(os.path.join(root, 'Test_labels.csv'), header=None)
        self.num = len(self.label)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = np.asarray(self.label, dtype=np.float)
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item].reshape(32, 20)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class MyDataNumpy:
    def __init__(self, root):
        super(MyDataNumpy, self).__init__()

        data_hy = pd.read_csv(os.path.join(root, 'data_HY.csv'), header=0)
        data_hy = np.asarray(data_hy, dtype=np.float)

        data_ls = pd.read_csv(os.path.join(root, 'data_LS.csv'), header=0)
        data_ls = np.asarray(data_ls, dtype=np.float)

        data_wt = pd.read_csv(os.path.join(root, 'data_WT.csv'), header=0)
        data_wt = np.asarray(data_wt, dtype=np.float)

        data_all = np.vstack((data_hy, data_ls, data_wt))
        np.random.shuffle(data_all)
        label = data_all[:, -2:]
        data = data_all[:, :-2]

        len_all = len(data_all)
        self.len_train = int(0.9 * len_all)
        self.len_test = len_all - self.len_train

        self.data_train = data[: self.len_train]
        self.data_test = data[self.len_train:]
        self.label_train = label[: self.len_train]
        self.label_test = label[self.len_train:]


class MyData(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(MyData, self).__init__()

        if train:
            self.data = pd.read_csv(os.path.join(root, 'data_train.csv'), header=0)
            self.label = pd.read_csv(os.path.join(root, 'label_train.csv'), header=0)
        else:
            self.data = pd.read_csv(os.path.join(root, 'data_test.csv'), header=0)
            self.label = pd.read_csv(os.path.join(root, 'label_test.csv'), header=0)
        self.num = len(self.label)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = np.asarray(self.label, dtype=np.float)
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item].reshape(8, 32)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
            data = data.type(torch.FloatTensor)
            label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class CNN(nn.Module):
    def __init__(self, dropout_rate):
        super(CNN, self).__init__()

        # input: 1 * 32 * 20
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
        )                                   # 32 * 32 * 20

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )                                   # 32 * 32 * 20

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )                                   # 64 * 16 * 10

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )                                   # 64 * 14 * 8

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )                                   # 64 * 14 * 8

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )                                   # 128 * 7 * 4

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 7 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(torch.cat((conv1, conv2), 1))
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(torch.cat((conv4, conv5), 1))
        conv6 = conv6.view(-1, 128 * 7 * 4)
        fc1 = self.fc1(conv6)
        out = self.fc2(fc1)
        return out


class CNN_EEG(nn.Module):
    def __init__(self, dropout_rate):
        super(CNN_EEG, self).__init__()

        # input: 1 * 8 * 32
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
        )                                   # 32 * 8 * 32

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )                                   # 32 * 8 * 32

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )                                   # 64 * 4 * 16

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )                                   # 64 * 4 * 16

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )                                   # 64 * 4 * 16

        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )                                   # 128 * 2 * 8

        self.fc1 = nn.Sequential(
            nn.Linear(128 * 2 * 8, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(torch.cat((conv1, conv2), 1))
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(torch.cat((conv4, conv5), 1))
        conv6 = conv6.view(-1, 128 * 2 * 8)
        fc1 = self.fc1(conv6)
        out = self.fc2(fc1)
        return out
