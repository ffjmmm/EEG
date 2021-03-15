import os
import torch
from torch import nn
import pandas as pd
import numpy as np


class EEGBCI_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None):
        super(EEGBCI_Dataset, self).__init__()

        if train:
            self.data = pd.read_csv(os.path.join(root, 'train_data.csv'), engine='python')
        else:
            self.data = pd.read_csv(os.path.join(root, 'test_data.csv'), engine='python')
        self.num = len(self.data)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = self.data[:, -2:]
        self.data = self.data[:, :-2]
        self.transform = transform

    def __getitem__(self, item):
        data = self.data[item].reshape(32, 20)
        data = torch.from_numpy(data)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class EEG_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, subject, transform=None):
        super(EEG_Dataset, self).__init__()

        if train:
            self.data = pd.read_csv(os.path.join(root, 'train_data_%s.csv' % subject), engine='python')
        else:
            self.data = pd.read_csv(os.path.join(root, 'test_data_%s.csv' % subject), engine='python')
        self.num = len(self.data)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = self.data[:, -2:]
        self.data = self.data[:, :-2]
        self.transform = transform
        print(self.data.shape)
        print(self.label.shape)

    def __getitem__(self, item):
        data = self.data[item].reshape(32, 32)
        data = torch.from_numpy(data)
        label = torch.from_numpy(self.label[item])
        if self.transform:
            data = self.transform(data)
        data = data.type(torch.FloatTensor)
        label = label.type(torch.FloatTensor)
        return data, label

    def __len__(self):
        return self.num


class EEG_Transfer_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train, subject, source=True, transform=None):
        super(EEG_Transfer_Dataset, self).__init__()

        root = os.path.join(root, subject)
        print(root)

        if train:
            if source:
                self.data = pd.read_csv(os.path.join(root, 'train_data_except.csv'), engine='python')
            else:
                self.data = pd.read_csv(os.path.join(root, 'train_data_subject.csv'), engine='python')
        else:
            if source:
                self.data = pd.read_csv(os.path.join(root, 'test_data_except.csv'), engine='python')
            else:
                self.data = pd.read_csv(os.path.join(root, 'test_data_subject.csv'), engine='python')

        self.num = len(self.data)
        self.data = np.asarray(self.data, dtype=np.float)
        self.label = self.data[:, -2:]
        self.data = self.data[:, :-2]
        self.transform = transform
        print(self.data.shape)
        print(self.label.shape)

    def __getitem__(self, item):
        data = self.data[item].reshape(32, 32)
        data = torch.from_numpy(data)
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

        # input 1 x 32 x 20
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

        # 64 * 16 * 10
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

        # 128 * 8 * 5
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 40, 512),
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
        x = x.view(-1, 1, 32, 20)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 128 * 40)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class EEG_CNN(nn.Module):
    def __init__(self, dropout_rate=0.0, transfer=False):
        super(EEG_CNN, self).__init__()

        # input 1 x 32 x 32
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

        # 64 * 16 * 16
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

        if transfer:
            for param in self.parameters():
                param.requires_grad = False

        # 128 * 8 * 8
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 64, 512),
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
        x = x.view(-1, 1, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 128 * 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class EEGBCI_CNN_RNN(nn.Module):
    def __init__(self, dropout_rate):
        super(EEGBCI_CNN_RNN, self).__init__()

        # input 1 x 8 x 20
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )
        # 64 * 4 * 10

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.conv4_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )


        # 64 * 16 * 10
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2),
        )

        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )

        # 128 * 8 * 5
        self.fc1 = nn.Sequential(
            nn.Linear(1408, 512),
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
        x = x.view(-1, 640)
        x_1 = x[:, 0: 160]
        x_2 = x[:, 160: 320]
        x_3 = x[:, 320: 480]
        x_4 = x[:, 480: 640]
        x_1 = x_1.view(-1, 1, 8, 20)
        x_2 = x_2.view(-1, 1, 8, 20)
        x_3 = x_3.view(-1, 1, 8, 20)
        x_4 = x_4.view(-1, 1, 8, 20)

        x_1 = self.conv1_1(x_1)
        x_1 = self.conv1_2(x_1)
        x_1 = self.conv1_3(x_1)

        x_2 = self.conv2_1(x_2)
        x_2 = self.conv2_2(x_2)
        x_2 = self.conv2_3(x_2)

        x_3 = self.conv3_1(x_3)
        x_3 = self.conv3_2(x_3)
        x_3 = self.conv3_3(x_3)

        x_4 = self.conv4_1(x_4)
        x_4 = self.conv4_2(x_4)
        x_4 = self.conv4_3(x_4)
        # BATCH * 64 * 4 * 10

        # BATCH * 64 * 16 * 10
        x = torch.cat((x_1, x_2, x_3, x_4), 2)

        x_cnn = self.conv4(x)
        x_cnn = self.conv5(x)
        # BATCH * 32 * 8 * 5

        x_rnn = x.view(-1, 64, 16 * 10)
        x_rnn = x_rnn.transpose(1, 2)
        # BATCH * 160 * 64
        x_rnn, h_n = self.rnn(x_rnn)
        x_rnn = x_rnn[:, -1, :]

        x_cnn = x_cnn.view(-1, 32 * 40)
        x_rnn = x_rnn.view(-1, 128)
        x = torch.cat((x_cnn, x_rnn), 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class EEGBCI_RNN(nn.Module):
    def __init__(self, dropout_rate):
        super(EEGBCI_RNN, self).__init__()

        # input 20 x 32
        self.rnn = nn.GRU(
            input_size=32,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out, h_n = self.rnn(x, None)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        out = self.fc3(out)
        return out