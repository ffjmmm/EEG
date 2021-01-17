import mne
import sklearn
import sklearn.model_selection
import sklearn.discriminant_analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils

tmin, tmax = -1., 4.
# train_interval = [(1.0, 1.5), (1.5, 2.0)]
num_channel = 4
len_data = 160 * 5
num_subjects = 109

# ch_names = ['C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1', 'P3', 'P1']
ch_names = ['C3', 'C1', 'CP1', 'P1']

mne.set_log_file('./mne_output', overwrite=True)

train_dict = {i: [] for i in range(num_channel * len_data)}
train_dict['left'] = []
train_dict['right'] = []
test_dict = {i: [] for i in range(num_channel * len_data)}
test_dict['left'] = []
test_dict['right'] = []

for subject_id in range(1, num_subjects + 1):
# for subject_id in range(89, 90):
    filenames = mne.datasets.eegbci.load_data(subject_id, [4, 8, 12], path='./datasets/eegbci')

    raw = mne.concatenate_raws([mne.io.read_raw_edf(filename, preload=True) for filename in filenames])
    mne.datasets.eegbci.standardize(raw)
    if raw.info['sfreq'] != 160:
        print('%d !!! sfreq not equals to 160' % subject_id)
        continue

    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    raw = raw.pick_channels(ch_names)
    print(raw)

    raw.filter(7., 30.)

    events, _ = mne.events_from_annotations(raw, event_id={'T1': 2, 'T2': 3})
    event_id = {'left': 2, 'right': 3}

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax - 1. / 160, proj=True, picks=mne.pick_types(raw.info, eeg=True),
                        baseline=None, preload=True)

    # ica = mne.preprocessing.ICA(n_components=0.95, method='fastica').fit(epochs)

    # ecg_epochs = mne.preprocessing.create_eog_epochs

    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 2

    csp = mne.decoding.CSP(n_components=4, log=True)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

    cross_validation = sklearn.model_selection.ShuffleSplit(1, test_size=0.1)
    # index_generator = cross_validation.split(epochs_data_train)
    index_generator = cross_validation.split(epochs_data)

    for train_index, test_index in index_generator:
        data_train = epochs_data[train_index]
        data_test = epochs_data[test_index]
        label_train = labels[train_index]
        label_test = labels[test_index]

        # data_train = csp.fit_transform(data_train, label_train)
        # data_test = csp.transform(data_test)
        #
        # lda.fit(data_train, label_train)
        # print(lda.score(data_test, label_test))
        # continue

        # Normalization / Standardization
        data_train = utils.preprocess(data_train, 'Normalization')
        data_test = utils.preprocess(data_test, 'Normalization')

        for i in range(len(data_train)):
            for j in range(num_channel * len_data):
                train_dict[j].append(data_train[i][j])
            train_dict['left'].append(label_train[i] ^ 1)
            train_dict['right'].append(label_train[i])

        for i in range(len(data_test)):
            for j in range(num_channel * len_data):
                test_dict[j].append(data_test[i][j])
            test_dict['left'].append(label_test[i] ^ 1)
            test_dict['right'].append(label_test[i])

print('Start saving data to csv file ...')
dataframe_train = pd.DataFrame(train_dict)
dataframe_train.to_csv('./datasets/train_data.csv', index=False)

dataframe_test = pd.DataFrame(test_dict)
dataframe_test.to_csv('./datasets/test_data.csv', index=False)
