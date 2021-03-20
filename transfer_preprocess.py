import scipy.io as sio
import numpy as np
import pandas as pd
import mne
import random

mne.set_log_file('./mne_output', overwrite=True)

srate = 1024
factor = 1. / 1000000.
interval = 5
start_time = 0
n_pnts = 1024
n_channels = 28
data_standardization = True
raw_plot = False
l_freq, h_freq = 5., 30.
channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
            'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', 'F7', 'F5', 'F3', 'F1', 'F2',
            'F4', 'F6', 'F8']
lab_index = 3
subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh']
subjects_num = 11
# subject = [1]
ch_names = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3',  'Fz',  'F4',  'F8',  'FC5', 'FC1', 'FC2', 'FC6',
            'M1',  'T7',  'C3',  'Cz', 'C4',  'T8',  'M2',  'CP5', 'CP1', 'CP2', 'CP6', 'P7',
            'P3',  'Pz',  'P4',  'P8', 'POz', 'O1',  'O2',  'EOG', 'AF7', 'AF3', 'AF4', 'AF8',
            'F5',  'F1',  'F2',  'F6', 'FC3', 'FCz', 'FC4', 'C5',  'C1',  'C2',  'C6',  'CP3',
            'CP4', 'P5',  'P1',  'P2', 'P6',  'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7',
            'TP8', 'PO7', 'PO8', 'Oz']
ch_types = ['eeg'] * 31 + ['eog'] + ['eeg'] * 32


# open 'name''index'.mat
def load_file(name, index):
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
            # label.append(event_label - 1004)
            if event_label == 1004:
                label.append([1, 0])
            else:
                label.append([0, 1])
    return time, label, raw_data


# load data from given subject and channels
def load_data(subject_index, index, channels):
    event_time, event_label, raw_data = load_file(subjects[subject_index], index)
    raw_data = raw_data * factor
    event_num = len(event_label)
    print(subject_index, index, event_num)

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=1024)
    info['description'] = 'My dataset'
    raw = mne.io.RawArray(raw_data, info)
    raw = raw.pick_channels(channels)
    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge')
    if raw_plot:
        raw.plot()
    raw_data = raw.get_data()

    data_label = []
    for i in range(event_num):
        time = event_time[i] + srate * start_time
        if time + n_pnts * interval >= raw_data.shape[1]:
            continue
        sample_point = np.arange(time, time + n_pnts * interval, interval)

        data_label += [np.hstack((raw_data[j][sample_point], event_label[i])) for j in range(n_channels)]

    return data_label


for selected_id in range(11):
    data_label_except = []
    data_label_subject = []

    for subject_index in range(subjects_num):
        data_label_subject = load_data(subject_index, lab_index, channels)
        if subject_index == selected_id:
            data_label_subject += data_label_subject
        else:
            data_label_except += data_label_subject

    random.seed(27)
    random.shuffle(data_label_except)
    random.shuffle(data_label_subject)

    data_label_except = np.stack([data_label for data_label in data_label_except], 0)
    data_label_subject = np.stack([data_label for data_label in data_label_subject], 0)

    n_data_except = len(data_label_except)
    n_train_except = int(0.9 * n_data_except)
    n_data_subject = len(data_label_subject)
    n_train_subject = int(0.9 * n_data_subject)

    # standardization
    if data_standardization:
        for i in range(n_pnts):
            mean_data = np.mean(data_label_except[:, i])
            std_data = np.std(data_label_except[:, i])
            data_label_except[:, i] = (data_label_except[:, i] - mean_data) / std_data

            mean_data = np.mean(data_label_subject[:, i])
            std_data = np.std(data_label_subject[:, i])
            data_label_subject[:, i] = (data_label_subject[:, i] - mean_data) / std_data

    # 保存数据到csv文件
    train_dict_except = {i: [] for i in range(n_pnts)}
    train_dict_except['left'] = []
    train_dict_except['right'] = []
    test_dict_except = {i: [] for i in range(n_pnts)}
    test_dict_except['left'] = []
    test_dict_except['right'] = []

    train_dict_subject = {i: [] for i in range(n_pnts)}
    train_dict_subject['left'] = []
    train_dict_subject['right'] = []
    test_dict_subject = {i: [] for i in range(n_pnts)}
    test_dict_subject['left'] = []
    test_dict_subject['right'] = []

    for i in range(n_train_except):
        for j in range(n_pnts):
            train_dict_except[j].append(data_label_except[i, j])
        train_dict_except['left'].append(data_label_except[i, -2])
        train_dict_except['right'].append(data_label_except[i, -1])

    for i in range(n_train_except, n_data_except):
        for j in range(n_pnts):
            test_dict_except[j].append(data_label_except[i, j])
        test_dict_except['left'].append(data_label_except[i, -2])
        test_dict_except['right'].append(data_label_except[i, -1])

    for i in range(n_train_subject):
        for j in range(n_pnts):
            train_dict_subject[j].append(data_label_subject[i, j])
        train_dict_subject['left'].append(data_label_subject[i, -2])
        train_dict_subject['right'].append(data_label_subject[i, -1])

    for i in range(n_train_subject, n_data_subject):
        for j in range(n_pnts):
            test_dict_subject[j].append(data_label_subject[i, j])
        test_dict_subject['left'].append(data_label_subject[i, -2])
        test_dict_subject['right'].append(data_label_subject[i, -1])

    print('Start saving data to csv file ...')
    dataframe_train = pd.DataFrame(train_dict_except)
    dataframe_train.to_csv('./datasets/lab_%d/%s/train_data_except.csv' % (lab_index, subjects[selected_id]), index=False)
    dataframe_test = pd.DataFrame(test_dict_except)
    dataframe_test.to_csv('./datasets/lab_%d/%s/test_data_except.csv' % (lab_index, subjects[selected_id]), index=False)


    dataframe_train = pd.DataFrame(train_dict_subject)
    dataframe_train.to_csv('./datasets/lab_%d/%s/train_data_subject.csv' % (lab_index, subjects[selected_id]), index=False)
    dataframe_test = pd.DataFrame(test_dict_subject)
    dataframe_test.to_csv('./datasets/lab_%d/%s/test_data_subject.csv' % (lab_index, subjects[selected_id]), index=False)

    print('lab: %d subject: %s complete' % (lab_index, subjects[selected_id]))
