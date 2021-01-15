import mne
import sklearn
import sklearn.model_selection
import sklearn.discriminant_analysis
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils

tmin, tmax = -1., 4.
csp_components = 4
train_interval = [(1.0, 1.5), (1.5, 2.0)]
num_channel = 10
len_data = 80
num_subjects = 109

ch_names = ['C5', 'C3', 'C1', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1']

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

    # band-pass filter on raw
    raw.filter(7., 30.)

    events, _ = mne.events_from_annotations(raw, event_id={'T1': 2, 'T2': 3})
    event_id = {'left': 2, 'right': 3}

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=mne.pick_types(raw.info, eeg=True),
                        baseline=None, preload=True)

    for train_tmin, train_tmax in train_interval:
        epochs_train = epochs.copy().crop(tmin=train_tmin, tmax=train_tmax - 1. / 160)
        labels = epochs.events[:, -1] - 2

        epochs_data_train = epochs_train.get_data()

        cross_validation = sklearn.model_selection.ShuffleSplit(1, test_size=0.1)
        index_generator = cross_validation.split(epochs_data_train)

        for train_index, test_index in index_generator:
            data_train = epochs_data_train[train_index]
            data_test = epochs_data_train[test_index]
            label_train = labels[train_index]
            label_test = labels[test_index]

            # Normalization / Standardization
            data_train = utils.preprocess(data_train, 'Standardization')
            data_test = utils.preprocess(data_test, 'Standardization')

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

exit()


lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
csp = mne.decoding.CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)

# csp.fit_transform(epochs_data, label)
# csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

for train_index, test_index in index_generator:
    label_train, label_text = labels[train_index], labels[test_index]
    print(epochs_data_train[train_index].shape)
    print(epochs_data_train[test_index].shape)
    data_train = csp.fit_transform(epochs_data_train[train_index], label_train)
    data_test = csp.transform(epochs_data_train[test_index])

    lda.fit(data_train, label_train)
    print('score: %f' % lda.score(data_test, label_text))


