import torch
from torch.utils.data import DataLoader

import utils

BATCH_SIZE = 64
DROPOUT_RATE = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
NEURAL_NETWORK_TYPE = 'CNN'
DATASET = 'EEG'

subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh', 'all']
subject = 11

print('Loading data ...')
dataset_test = utils.EEG_Dataset(root='./datasets', train=False, subject=subjects[subject])

dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, drop_last=True)

print('Loading Neural Network ...')
neural_network = utils.EEG_CNN(DROPOUT_RATE)
neural_network.load_state_dict(torch.load('./models/' + DATASET + '_' + NEURAL_NETWORK_TYPE + '.pkl'))
neural_network.to(device)

print('Starting test ...')
neural_network.eval()
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

print('test accuracy: %.4f' % avg_test_accuracy)
