import scipy.io as sio
import numpy as np
import pandas as pd
import mne
import random

mne.set_log_file('./mne_output', overwrite=True)

srate = 1024
resample_rate = 512
factor = 1. / 1000000.
interval = 5
start_time = 1
n_pnts = 1024 * 20

# 'Standardization' / 'Normalization' / 'None'
preprocess_method = 'Standardization'

raw_plot = False
epoch_plot = False
l_freq, h_freq = 2., 36.
# channels = ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6',
#             'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', 'F7', 'F5', 'F3', 'F1', 'F2',
#             'F4', 'F6', 'F8']
channels = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3',  'C1',
            'Cz',  'C2',  'C4',  'C6',  'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6']

subjects = ['ckm', 'clx', 'csb', 'fy', 'lw', 'ly', 'phl', 'szl', 'xwt', 'yfw', 'zjh']
subjects_num = 11
# subject_id_selected = [i for i in range(subjects_num)]
subject_id_selected = [0]
lab_index = 1
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
            label.append(event_label)
    return time, label, raw_data * factor


# load data from given subject and channels
def load_data(subject_index, index, channels):
    event_time, event_label, raw_data = load_file(subjects[subject_index], index)
    event_num = len(event_label)
    event_duration = np.ones(event_num, dtype=np.int) * 4
    events = np.column_stack((event_time, event_duration, event_label))

    print(subject_index, index, event_num)

    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=srate)
    info['description'] = 'My dataset'
    info.set_montage('standard_1005')

    raw = mne.io.RawArray(raw_data, info)
    annotations = mne.annotations_from_events(events, srate, event_desc={1004: 'left', 1005: 'right'})
    raw.set_annotations(annotations)

    raw = raw.pick_channels(channels)

    raw = raw.resample(resample_rate)

    raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge')
    if raw_plot:
        raw.plot()

    events, event_id = mne.events_from_annotations(raw)
    reject_criteria = dict(eeg=150e-6)
    epochs = mne.Epochs(raw, events, tmin=-0.1, tmax=3.9 - 1. / resample_rate, event_id=event_id, reject=reject_criteria, preload=True)
    if epoch_plot:
        epochs.plot()
    # print(epochs)

    epoch_data = epochs.get_data()
    events = epochs.events
    event_num = len(events)

    data_label = []
    for i in range(event_num):
        for j in range(2 * resample_rate):
            data_ = epoch_data[i, :, j: j + 2 * resample_rate].reshape(-1)
            label_ = [abs(events[i][-1] - 2), events[i][-1] - 1]
            data_label += [np.hstack((data_, label_))]

    print(len(data_label), len(data_label[0]))

    return data_label

    # raw_data = raw.get_data()
    #
    # data_label = []
    # for i in range(event_num):
    #     time = event_time[i] + srate * start_time
    #     if time + n_pnts * interval >= raw_data.shape[1]:
    #         continue
    #     sample_point = np.arange(time, time + n_pnts * interval, interval)
    #
    #     data_label += [np.hstack((raw_data[j][sample_point], event_label[i])) for j in range(n_channels)]


data_label_all = []

for subject_index in subject_id_selected:
    data_label_all += load_data(subject_index, lab_index, channels)

random.seed(27)
random.shuffle(data_label_all)
data_label_all = np.stack([data_label for data_label in data_label_all], 0)

n_data = len(data_label_all)
n_train = int(0.9 * n_data)

# preprocess
if preprocess_method == 'Standardization':
    for i in range(n_pnts):
        mean_data = np.mean(data_label_all[:, i])
        std_data = np.std(data_label_all[:, i])
        data_label_all[:, i] = (data_label_all[:, i] - mean_data) / std_data
elif preprocess_method == 'Normalization':
    for i in range(n_pnts):
        min_data = np.min(data_label_all[:, i])
        max_data = np.max(data_label_all[:, i])
        data_label_all[:, i] = (data_label_all[:, i] - min_data) / (max_data - min_data)


# 保存数据到csv文件
train_dict = {i: [] for i in range(n_pnts)}
train_dict['left'] = []
train_dict['right'] = []
test_dict = {i: [] for i in range(n_pnts)}
test_dict['left'] = []
test_dict['right'] = []

for i in range(n_train):
    for j in range(n_pnts):
        train_dict[j].append(data_label_all[i, j])
    train_dict['left'].append(data_label_all[i, -2])
    train_dict['right'].append(data_label_all[i, -1])

for i in range(n_train, n_data):
    for j in range(n_pnts):
        test_dict[j].append(data_label_all[i, j])
    test_dict['left'].append(data_label_all[i, -2])
    test_dict['right'].append(data_label_all[i, -1])

print('Start saving data to csv file ...')
dataframe_train = pd.DataFrame(train_dict)
filename = '%s.csv' % subjects[subject_id_selected[0]] if len(subject_id_selected) == 1 else 'all.csv'
dataframe_train.to_csv('./datasets/train_data_' + filename, index=False)

dataframe_test = pd.DataFrame(test_dict)
dataframe_test.to_csv('./datasets/test_data_' + filename, index=False)

