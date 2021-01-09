import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np
import shutil
import os

import utils


EPOCH = 200
INPUT_SIZE = 8 * 32
N_CLASS = 2
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
LR = 0.001
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if os.path.exists('runs/ECG_LR=0.001_DROP=0.5'):
    shutil.rmtree('runs/ECG_LR=0.001_DROP=0.5')
writer = SummaryWriter('runs/ECG_LR=0.001_DROP=0.5')

print('Loading data ...')
train_dataset = utils.MyData(root='./dataset', train=True, transform=transforms.ToTensor())
test_dataset = utils.MyData(root='./dataset', train=False, transform=transforms.ToTensor())
# print(train_dataset.num)
# print(test_dataset.num)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

print('Constructing CNN ...')
cnn = utils.CNN_EEG(DROPOUT_RATE).to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

print('Start Training ...')
for epoch in range(EPOCH):
    cnn = cnn.train()
    avg_loss = 0
    for step, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        output = cnn(data)
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().data.numpy()
    avg_loss /= (step + 1)

    cnn = cnn.eval()
    avg_accuracy = 0.0
    for step, (data, label) in enumerate(test_loader):
        data = data.to(device)
        output = cnn(data).cpu()
        pred = torch.max(output, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()

        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        avg_accuracy += accuracy
    avg_accuracy /= (step + 1)
    print('Epoch: ', epoch, '| train loss: %.5f' % avg_loss, '| accuracy: %.5f' % avg_accuracy)
    writer.add_scalar('loss', avg_loss, epoch)
    writer.add_scalar('accuracy', avg_accuracy, epoch)
