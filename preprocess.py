import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

srate = 1024
interval = 5
n_pnts = 1024
n_channels = 64
channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
            'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', 'F7', 'F5', 'F3', 'F1', 'F2',
            'F4', 'F6', 'F8']
subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh']
subjects_num = 11
ELECTRODES = {
    'Fp1': 1, 'Fpz': 2, 'Fp2': 3, 'F7': 4, 'F3': 5, 'Fz': 6, 'F4': 7, 'F8': 8, 'FC5': 9, 'FC1': 10,
    'FC2': 11, 'FC6': 12, 'M1': 13, 'T7': 14, 'C3': 15, 'Cz': 16, 'C4': 17, 'T8': 18, 'M2': 19, 'CP5': 20,
    'CP1': 21, 'CP2': 22, 'CP6': 23, 'P7': 24, 'P3': 25, 'Pz': 26, 'P4': 27, 'P8': 28, 'POz': 29, 'O1': 30,
    'O2': 31, 'EOG': 32, 'AF7': 33, 'AF3': 34, 'AF4': 35, 'AF8': 36, 'F5': 37, 'F1': 38, 'F2': 39, 'F6': 40,
    'FC3': 41, 'FCz': 42, 'FC4': 43, 'C5': 44, 'C1': 45, 'C2': 46, 'C6': 47, 'CP3': 48, 'CP4': 49, 'P5': 50,
    'P1': 51, 'P2': 52, 'P6': 53, 'PO5': 54, 'PO3': 55, 'PO4': 56, 'PO6': 57, 'FT7': 58, 'FT8': 59, 'TP7': 60,
    'TP8': 61, 'PO7': 62, 'PO8': 63, 'Oz': 64,
}


# 读取文件，获取其raw数据与events
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
            if event_label == 1004:
                label.append([1, 0])
            else:
                label.append([0, 1])
    return time, label, raw_data


# 读取subject index给定channels的所有数据
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
            data_ = raw_data[ELECTRODES[channel] - 1][sample_point]
            # data_ = raw_data[channel - 1][sample_point]
            data.append(data_)
            labels.append(event_label[i])

    return data, labels


data_all = []
labels_all = []

# 读取所有数据
for subject_index in range(subjects_num):
    data_ = []
    labels_ = []
    for k in range(3):
        data, labels = load_data(subject_index, k, channels)
        data_ += data
        labels_ += labels
        print(subject_index, k, len(data))
    for i in range(1, len(data_)):
        data_[0] = np.vstack((data_[0], data_[i]))
        labels_[0] = np.vstack((labels_[0], labels_[i]))
    data_all.append(data_[0])
    labels_all.append(labels_[0])


# 将所有数据堆叠在一起
total = len(data_all)
for i in range(1, len(data_all)):
    data_all[0] = np.vstack((data_all[0], data_all[i]))
    labels_all[0] = np.vstack((labels_all[0], labels_all[i]))
    print(i, total)

data = data_all[0]
labels = labels_all[0]

print(data.shape)
print(labels.shape)

# 进行数据标准化
for i in range(data.shape[1]):
    mean_data = np.mean(data[:, i])
    std_data = np.std(data[:, i])
    data[:, i] = (data[:, i] - mean_data) / std_data

# 打乱数据
data_labels = np.hstack((data, labels))
np.random.shuffle(data_labels)

# 保存数据到csv文件
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

