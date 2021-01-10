import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import utils


EPOCH = 100
BATCH_SIZE = 32
INPUT_SIZE = 10
N_CLASS = 2
LR = 0.01
DROPOUT_RATE = 0.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter()

if os.path.exists('runs/ECG_LR=%.3f_DROP=%.2f' % (LR, DROPOUT_RATE)):
    shutil.rmtree('runs/ECG_LR=%.3f_DROP=%.2f' % (LR, DROPOUT_RATE))
writer = SummaryWriter('runs/ECG_LR=%.3f_DROP=%.2f' % (LR, DROPOUT_RATE))

print('Loading data ...')
dataset_train = utils.EEGBCI_Dataset(root='./', train=True, transform=transforms.ToTensor())
dataset_test = utils.EEGBCI_Dataset(root='./', train=False, transform=transforms.ToTensor())
# print(train_dataset.num)
# print(test_dataset.num)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE)

print('Constructing Neural Network ...')
fc = utils.EEGBCI_FC(DROPOUT_RATE).to(device)
optimizer = torch.optim.Adam(fc.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()

print('Start Training ...')
for epoch in range(EPOCH):
    fc = fc.train()
    avg_loss = 0.
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        label = label.type(torch.long).reshape(-1)
        out = fc(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.cpu().data.numpy()
    avg_loss /= (step + 1)

    fc = fc.eval()
    avg_accuracy = 0.0
    for step, (data, label) in enumerate(dataloader_test):
        data = data.to(device)
        out = fc(data).cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = label.numpy().reshape(-1)
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        avg_accuracy += accuracy
    avg_accuracy /= (step + 1)
    print('Epoch: ', epoch, '| train loss: %.4f' % avg_loss, '| accuracy: %.2f' % avg_accuracy)
    writer.add_scalar('loss', avg_loss, epoch)
    writer.add_scalar('accuracy', avg_accuracy, epoch)