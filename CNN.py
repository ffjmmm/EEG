import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import utils


EPOCH = 100
BATCH_SIZE = 16
INPUT_CHANNEL = 4
INPUT_LEN = 800
N_CLASS = 2
LR = 0.0001
DROPOUT_RATE = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEURAL_NETWORK_TYPE = 'CNN'


if os.path.exists('runs/ECG_LR=%.4f' % LR):
    shutil.rmtree('runs/ECG_LR=%.4f' % LR)
writer = SummaryWriter('runs/ECG_LR=%.4f' % LR)

print('Loading data ...')
dataset_train = utils.EEGBCI_Dataset(root='./datasets', train=True, transform=transforms.ToTensor())
dataset_test = utils.EEGBCI_Dataset(root='./datasets', train=False, transform=transforms.ToTensor())
# print(train_dataset.num)
# print(test_dataset.num)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)

print('Constructing Neural Network ...')
neural_network = None
if NEURAL_NETWORK_TYPE == 'CNN':
    neural_network = utils.EEGBCI_CNN(DROPOUT_RATE).to(device)
else:
    neural_network = utils.EEGBCI_RNN(DROPOUT_RATE).to(device)
optimizer = torch.optim.Adam(neural_network.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

print('Start Training ...')
for epoch in range(EPOCH):
    neural_network = neural_network.train()
    avg_train_loss = 0.
    avg_train_accuracy = 0.0
    for step, (data, label) in enumerate(dataloader_train):
        # data = data.numpy().reshape(-1, 10, 80)
        # x = np.linspace(0, 79, 80)
        # for i in range(5):
        #     plt.clf()
        #     for j in range(5):
        #         plt.plot(x, data[i][j], label='%d' % j)
        #     plt.title(label.numpy()[i])
        #     plt.legend()
        #     plt.show()
        # exit()

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

    avg_train_loss /= (step + 1)
    avg_train_accuracy /= (step + 1)

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

    avg_test_accuracy /= (step + 1)
    avg_test_loss /= (step + 1)

    print('Epoch: ', epoch, '| train loss: %.4f' % avg_train_loss, '| train accuracy: %.2f' % avg_train_accuracy,
          '| test loss: %.4f' % avg_test_loss, '| test accuracy: %.2f' % avg_test_accuracy)
    writer.add_scalar('loss', avg_train_loss, epoch)
    writer.add_scalar('train_accuracy', avg_train_accuracy, epoch)
    writer.add_scalar('test_accuracy', avg_test_accuracy, epoch)