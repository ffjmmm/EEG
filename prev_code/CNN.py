import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import numpy as np

import utils


EPOCH = 1000
INPUT_SIZE = 32 * 20
N_CLASS = 4
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
LR = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()

print('Loading data ...')
train_dataset = utils.PhysioNetDataset(root='./dataset', train=True, transform=transforms.ToTensor())
test_dataset = utils.PhysioNetDataset(root='./dataset', train=False, transform=transforms.ToTensor())
# print(train_dataset.num)
# print(test_dataset.num)
# 134487
# 14943
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

print('Constructing CNN ...')
cnn = utils.CNN(DROPOUT_RATE).to(device)
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
    print('Epoch: ', epoch, '| train loss: %.4f' % avg_loss, '| accuracy: %.2f' % avg_accuracy)
    writer.add_scalar('loss', avg_loss, epoch)
    writer.add_scalar('accuracy', avg_accuracy, epoch)
