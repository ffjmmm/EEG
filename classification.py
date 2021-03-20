import os
import shutil
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils


EPOCH = 50
BATCH_SIZE = 16
LR = 0.001
DROPOUT_RATE = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEURAL_NETWORK_TYPE = 'CNN'
DATASET = 'EEG'
SUMMARY = True
SAVE_MODEL = False

subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh', 'all']
subject = 1
lab_index = 3

print('%s lab: %d LR: %.3f subject: %s start...' % (NEURAL_NETWORK_TYPE, lab_index, LR, subjects[subject]))

writer = None
if SUMMARY:
    if os.path.exists('runs/%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject])):
        shutil.rmtree('runs/%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject]))
    writer = SummaryWriter('runs/%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject]))

    # if os.path.exists('runs/source_%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject])):
    #     shutil.rmtree('runs/source_%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject]))
    # writer = SummaryWriter('runs/source_%s_lab%d_%s' % (NEURAL_NETWORK_TYPE, lab_index, subjects[subject]))


print('Loading data ...')
# dataset_train = utils.EEG_Dataset(root='./datasets/lab_%d' % lab_index, train=True, subject=subjects[subject]) if DATASET == 'EEG' else \
#     utils.EEGBCI_Dataset(root='./datasets', train=True, transform=None)
# dataset_test = utils.EEG_Dataset(root='./datasets/lab_%d' % lab_index, train=False, subject=subjects[subject]) if DATASET == 'EEG' else \
#     utils.EEGBCI_Dataset(root='./datasets', train=False, transform=None)
# dataset_train = utils.EEG_Transfer_Dataset(root='./datasets/lab_%d' % lab_index, train=True,
#                                            subject=subjects[subject], source=True)
# dataset_test = utils.EEG_Transfer_Dataset(root='./datasets/lab_%d' % lab_index, train=False,
#                                           subject=subjects[subject], source=True)
dataset_train = utils.EEG_Transfer_Dataset(root='./datasets/lab_%d' % lab_index, train=True,
                                           subject=subjects[subject], source=False)
dataset_test = utils.EEG_Transfer_Dataset(root='./datasets/lab_%d' % lab_index, train=False,
                                          subject=subjects[subject], source=False)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, drop_last=True)

print('Constructing Neural Network ...')
neural_network = None
if NEURAL_NETWORK_TYPE == 'CNN':
    neural_network = utils.EEG_CNN(DROPOUT_RATE).to(device) if DATASET == 'EEG' else utils.EEGBCI_CNN(DROPOUT_RATE).to(device)
# elif NEURAL_NETWORK_TYPE == 'RNN':
#     neural_network = utils.EEGBCI_RNN(DROPOUT_RATE).to(device)
# else:
#     neural_network = utils.EEGBCI_CNN_RNN(DROPOUT_RATE).to(device)

optimizer = torch.optim.Adam(neural_network.parameters(), lr=LR)
loss_func = torch.nn.MSELoss()

print('Start Training ...')
best_model_acc = 0.0
for epoch in range(EPOCH):
    neural_network = neural_network.train()
    avg_train_loss = 0.
    for step, (data, label) in enumerate(dataloader_train):
        data, label = data.to(device), label.to(device)
        out = neural_network(data)
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.cpu().data.numpy()

    avg_train_loss /= (step + 1)

    neural_network = neural_network.eval()
    avg_test_accuracy = 0.0
    for step, (data, label) in enumerate(dataloader_test):
        data, label = data.to(device), label.to(device)
        out = neural_network(data)

        out = out.cpu()
        label = label.cpu()
        pred = torch.max(out, 1)[1].data.numpy()
        label = torch.max(label, 1)[1].data.numpy()
        accuracy = float((pred == label).astype(int).sum()) / float(len(label))
        avg_test_accuracy += accuracy

    avg_test_accuracy /= (step + 1)

    print('Epoch: ', epoch, '| train loss: %.4f' % avg_train_loss, '| test accuracy: %.4f' % avg_test_accuracy)
    if SUMMARY:
        writer.add_scalar('loss', avg_train_loss, epoch)
        writer.add_scalar('accuracy', avg_test_accuracy, epoch)

    if SAVE_MODEL and avg_test_accuracy > best_model_acc:
        best_model_acc = avg_test_accuracy
        torch.save(neural_network.state_dict(), './models/%s_%s_lab%d.pkl' % (NEURAL_NETWORK_TYPE, subjects[subject], lab_index))
        print('Best model saved')
