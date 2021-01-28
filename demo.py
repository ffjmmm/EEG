import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

srate = 1024
interval = 10
n_pnts = 1000
n_channels = 64
subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh']
subjects_num = 11



def load_file(name, index):
    index += 1
    folder_path = './datasets/data/%s/' % name
    mat = sio.loadmat(folder_path + '%s%d.mat' % (name, index))
    raw_data = mat['EEG']['data'][0, 0]
    events = mat['EEG']['event'][0, 0][0]
    time = []
    label = []
    for i in range(len(events)):
        event_time, event_label = events[i][0][0, 0], int(events[i][1][0].split(',')[0])
        if event_label in [1004, 1005]:
            time.append(event_time)
            label.append(event_label - 1004)
    return time, label, raw_data


def load_data(subject_index, index, channels):
    event_time, event_label, raw_data = load_file(subjects[subject_index], index)
    event_num = len(event_label)

    data = []
    labels = []

    for i in range(event_num):
        time = event_time[i] - srate
        if time + n_pnts * interval >= raw_data.shape[1]:
            continue
        sample_point = np.arange(time, time + n_pnts * interval, interval)
        for channel in channels:
            data_ = raw_data[channel][sample_point]
            data.append(data_)
            labels.append(event_label[i])

    return data, labels


data_all = []
labels_all = []

for subject_index in range(subjects_num):
    for k in range(3):
        data, labels = load_data(subject_index, k, np.arange(n_channels))
        data_all += data
        labels_all += labels
        print(subject_index, k, len(data))

total = len(data_all)
for i in range(1, len(data_all)):
    data_all[0] = np.vstack((data_all[0], data_all[i]))
    labels_all[0] = np.vstack((labels_all[0], labels_all[i]))
    print(i, total)

data = data_all[0]
labels = labels_all[0]

print(data.shape)

for i in range(data.shape[1]):
    mean_data = np.mean(data[:, i])
    std_data = np.std(data[:, i])
    data[:, i] = (data[:, i] - mean_data) / std_data

data_labels = np.hstack((data, labels))
np.random.shuffle(data_labels)

train_dict = {i: [] for i in range(data.shape[1])}
train_dict['left'] = []
train_dict['right'] = []
test_dict = {i: [] for i in range(data.shape[1])}
test_dict['left'] = []
test_dict['right'] = []

num_data = len(data_labels)
num_train = int(0.9 * num_data)

for i in range(num_train):
    for j in range(data.shape[1]):
        train_dict[j].append(data_labels[i, j])
    train_dict['left'].append(data_labels[i, -2])
    train_dict['right'].append(data_labels[i, -1])

for i in range(num_train, num_data):
    for j in range(data.shape[1]):
        test_dict[j].append(data_labels[i, j])
    test_dict['left'].append(data_labels[i, -2])
    test_dict['right'].append(data_labels[i, -1])

print('Start saving data to csv file ...')
dataframe_train = pd.DataFrame(train_dict)
dataframe_train.to_csv('./datasets/train_data_new.csv', index=False)

dataframe_test = pd.DataFrame(test_dict)
dataframe_test.to_csv('./datasets/test_data_new.csv', index=False)

exit()
n = 1000000
m = 20
x = np.arange(0, n, interval)

for i in range(m):
    plt.plot(x, data[i][0: n: interval])
plt.show()
