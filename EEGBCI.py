import mne
import sklearn.model_selection
import pandas as pd

tmin, tmax = -1., 4.
csp_components = 10

mne.set_log_file('./mne_output', overwrite=True)

train_dict = {i: [] for i in range(csp_components)}
train_dict['label'] = []
test_dict = {i: [] for i in range(csp_components)}
test_dict['label'] = []

for subject_id in range(1, 110):
    filenames = mne.datasets.eegbci.load_data(subject_id, [4, 8, 12], path='./datasets/eegbci')
    raw = mne.concatenate_raws([mne.io.read_raw_edf(filename, preload=True) for filename in filenames])
    # print(raw.annotations.duration)

    mne.datasets.eegbci.standardize(raw)

    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # band-pass filter on raw
    raw.filter(7., 30.)

    events, _ = mne.events_from_annotations(raw, event_id={'T1': 2, 'T2': 3})
    event_id = {'left': 2, 'right': 3}

    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=mne.pick_types(raw.info, eeg=True),
                        baseline=None, preload=True)

    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    label = epochs.events[:, -1] - 2
    # print(epochs.event_id)
    # print(epochs.info)

    # epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()

    cross_validation = sklearn.model_selection.ShuffleSplit(1, test_size=0.1, random_state=27)
    index_generator = cross_validation.split(epochs_data_train)

    csp = mne.decoding.CSP(n_components=csp_components, reg=None, log=True, norm_trace=False)

    # csp.fit_transform(epochs_data, label)
    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    for train_index, test_index in index_generator:
        # print(train_index, test_index)
        label_train, label_text = label[train_index], label[test_index]
        data_train = csp.fit_transform(epochs_data_train[train_index], label_train)
        data_test = csp.transform(epochs_data_train[test_index])

        for i in range(data_train.shape[0]):
            for j in range(csp_components):
                train_dict[j].append(data_train[i][j])
            train_dict['label'].append(label_train[i])

        for i in range(data_test.shape[0]):
            for j in range(csp_components):
                test_dict[j].append(data_test[i][j])
            test_dict['label'].append(label_text[i])

dataframe_train = pd.DataFrame(train_dict)
dataframe_train.to_csv('./train_data.csv', index=False)

dataframe_test = pd.DataFrame(test_dict)
dataframe_test.to_csv('./test_data.csv', index=False)
