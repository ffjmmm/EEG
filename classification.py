import os
import shutil
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

import utils


# 超参数设置
EPOCH = 100
BATCH_SIZE = 128
N_CLASS = 2
LR = 0.0001
DROPOUT_RATE = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEURAL_NETWORK_TYPE = 'CNN'
SUMMARY = False


# 保存运行结果到tensorboard
writer = None
if SUMMARY:
    if os.path.exists('runs/EEG_' + NEURAL_NETWORK_TYPE + '_LR=%.4f' % LR):
        shutil.rmtree('runs/EEG_' + NEURAL_NETWORK_TYPE + '_LR=%.4f' % LR)
    writer = SummaryWriter('runs/EEG_' + NEURAL_NETWORK_TYPE + '_LR=%.4f' % LR)

print('Loading data ...')
# dataset_train = utils.EEGBCI_Dataset(root='./datasets', train=True, transform=None)
# dataset_test = utils.EEGBCI_Dataset(root='./datasets', train=False, transform=None)
dataset_train = utils.EEG_Dataset(root='./datasets', train=True, transform=None)
dataset_test = utils.EEG_Dataset(root='./datasets', train=False, transform=None)
# print(train_dataset.num)
# print(test_dataset.num)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, drop_last=True)
# (113414, 1024)
# (12602, 1024)

print('Constructing Neural Network ...')
neural_network = None
if NEURAL_NETWORK_TYPE == 'CNN':
    neural_network = utils.EEG_CNN(DROPOUT_RATE).to(device)
elif NEURAL_NETWORK_TYPE == 'RNN':
    neural_network = utils.EEGBCI_RNN(DROPOUT_RATE).to(device)
else:
    neural_network = utils.EEGBCI_CNN_RNN(DROPOUT_RATE).to(device)

optimizer = torch.optim.Adam(neural_network.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

print('Start Training ...')
for epoch in range(EPOCH):
    # 训练过程
    neural_network = neural_network.train()
    avg_train_loss = 0.
    avg_train_accuracy = 0.0
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        out = neural_network(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.cpu().data.numpy()

        out = out.cpu()
        label = label.cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        avg_train_accuracy += accuracy
    # 计算训练平均loss与accuracy
    avg_train_loss /= (step + 1)
    avg_train_accuracy /= (step + 1)

    # 测试过程
    neural_network = neural_network.eval()
    avg_test_loss = 0.0
    avg_test_accuracy = 0.0
    for step, (data, label) in enumerate(dataloader_test):
        data, label = data.to(device), label.to(device)
        out = neural_network(data)
        loss = loss_func(out, label)
        avg_test_loss += loss.cpu().data.numpy()

        out = out.cpu()
        label = label.cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        avg_test_accuracy += accuracy

    # 计算测试平均loss与accuracy
    avg_test_accuracy /= (step + 1)
    avg_test_loss /= (step + 1)

    print('Epoch: ', epoch, '| train loss: %.4f' % avg_train_loss, '| train accuracy: %.2f' % avg_train_accuracy,
          '| test loss: %.4f' % avg_test_loss, '| test accuracy: %.2f' % avg_test_accuracy)
    if SUMMARY:
        writer.add_scalar('loss', avg_train_loss, epoch)
        writer.add_scalar('accuracy', avg_test_accuracy, epoch)