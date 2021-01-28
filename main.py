import numpy as np
import io
import pyedflib
import pandas as pd


SAMPLE_RATE = 160
EEG_CHANNELS = 64
MI_RUNS = [4, 8, 12]
ELECTRODES = [8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 48, 49, 50, 52, 53]
RUN_LENGTH = 125 * SAMPLE_RATE
TRIAL_LENGHTH = 4

PHYSIONET_ELECTRODES = {
    1:  "FC5",  2: "FC3",  3: "FC1",  4: "FCz",  5: "FC2",  6: "FC4",
    7:  "FC6",  8: "C5",   9: "C3",  10: "C1",  11: "Cz",  12: "C2",
    13: "C4",  14: "C6",  15: "CP5", 16: "CP3", 17: "CP1", 18: "CPz",
    19: "CP2", 20: "CP4", 21: "CP6", 22: "Fp1", 23: "Fpz", 24: "Fp2",
    25: "AF7", 26: "AF3", 27: "AFz", 28: "AF4", 29: "AF8", 30: "F7",
    31: "F5",  32: "F3",  33: "F1",  34: "Fz",  35: "F2",  36: "F4",
    37: "F6",  38: "F8",  39: "FT7", 40: "FT8", 41: "T7",  42: "T8",
    43: "T9",  44: "T10", 45: "TP7", 46: "TP8", 47: "P7",  48: "P5",
    49: "P3",  50: "P1",  51: "Pz",  52: "P2",  53: "P4",  54: "P6",
    55: "P8",  56: "PO7", 57: "PO3", 58: "POz", 59: "PO4", 60: "PO8",
    61: "O1",  62: "Oz",  63: "O2",  64: "Iz"
}


def projection_2d(loc):
    """
    Azimuthal equidistant projection (AEP) of 3D carthesian coordinates.
    Preserves distance to origin while projecting to 2D carthesian space.
    loc: N x 3 array of 3D points
    returns: N x 2 array of projected 2D points
    """
    x, y, z = loc[:, 0], loc[:, 1], loc[:, 2]
    theta = np.arctan2(y, x)  # theta = azimuth
    rho = np.pi / 2 - np.arctan2(z, np.hypot(x, y))  # rho = pi/2 - elevation
    return np.stack((np.multiply(rho, np.cos(theta)), np.multiply(rho, np.sin(theta))), 1)


def get_electrode_positions():
    """
    Returns a dictionary (Name) -> (x,y,z) of electrode name in the extended
    10-20 system and its carthesian coordinates in unit sphere.
    """
    positions = dict()
    with io.open("./datasets/electrode_positions.txt", "r") as pos_file:
        for line in pos_file:
            parts = line.split()
            positions[parts[0]] = tuple([float(part) for part in parts[1:]])
    return np.array([positions[PHYSIONET_ELECTRODES[idx]] for idx in range(1, 65)])


def load_edf_signals(path):
    sig = pyedflib.EdfReader(path)
    n = sig.signals_in_file
    sigbuf = np.zeros((n, sig.getNSamples()[0]))

    for i in range(n):
        sigbuf[i, :] = sig.readSignal(i)
    annotations = sig.read_annotation()

    return sigbuf.transpose(), annotations


def load_data(subject):
    raw = np.zeros((3, RUN_LENGTH, EEG_CHANNELS))
    events = []

    base_path = './datasets/eegbci/MNE-eegbci-data/files/eegmmidb/1.0.0/S%03d/S%03dR%02d.edf'

    for i, current in enumerate(MI_RUNS):
        path = base_path % (subject, subject, current)
        signals, annotations = load_edf_signals(path)
        raw[i, :signals.shape[0], :] = signals

        current_event = [i, 0, 0, 0]

        for annotation in annotations:
            t = int(annotation[0] * SAMPLE_RATE * 1e-7)
            action = int(annotation[2][1])

            if action == 0 and current_event[1] != 0:
                length = TRIAL_LENGHTH * SAMPLE_RATE
                pad = (length - (t - current_event[2])) / 2
                current_event[2] -= pad + (t - current_event[2]) % 2
                current_event[3] = t + pad

                events.append(current_event)
            elif action > 0:
                current_event = [i, action, t, 0]

    data = np.zeros((len(events), TRIAL_LENGHTH * SAMPLE_RATE, EEG_CHANNELS))
    labels = np.zeros((len(events), 2))

    for i, event in enumerate(events):
        data[i, :, :] = raw[event[0], event[2]:event[3]]
        labels[i, event[1] - 1] = 1.

    return data, labels, projection_2d(get_electrode_positions())


def load_raw_data(electrodes, subjects):
    data = []
    labels = []

    for subject in subjects:
        data_subject, labels_subject, loc = load_data(subject)
        if data_subject.shape[0] != 42:
            print(subject, '!!!!!!')
            continue
        data.append(data_subject[:, :, electrodes])
        labels.append(labels_subject)

    return np.array(data, dtype=np.float64).squeeze(), np.array(labels, dtype=np.float64)

data_all = []
labels_all = []

for electrode in ELECTRODES:
    print('Electrode ' + PHYSIONET_ELECTRODES[electrode] + ' ...')
    data_electrode, labels_electrode = load_raw_data([electrode], range(1, 110))
    data_electrode = data_electrode.reshape(-1, 640)
    labels_electrode = labels_electrode.reshape(-1, 2)
    data_all.append(data_electrode)
    labels_all.append(labels_electrode)
#
# for i in range(1, len(data_all)):
#     data_all[0] = np.vstack((data_all[0], data_all[i]))
#     labels_all[0] = np.vstack((labels_all[0], labels_all[i]))
#
# data = data_all[0]
# labels = labels_all[0]
#
# for i in range(640):
#     mean_data = np.mean(data[:, i])
#     std_data = np.std(data[:, i])
#     data[:, i] = (data[:, i] - mean_data) / std_data
#
# data_labels = np.hstack((data, labels))
# np.random.shuffle(data_labels)
#
# train_dict = {i: [] for i in range(640)}
# train_dict['left'] = []
# train_dict['right'] = []
# test_dict = {i: [] for i in range(640)}
# test_dict['left'] = []
# test_dict['right'] = []
#
# num_data = len(data_labels)
# num_train = int(0.9 * num_data)
#
# for i in range(num_train):
#     for j in range(640):
#         train_dict[j].append(data_labels[i, j])
#     train_dict['left'].append(data_labels[i, -2])
#     train_dict['right'].append(data_labels[i, -1])
#
# for i in range(num_train, num_data):
#     for j in range(640):
#         test_dict[j].append(data_labels[i, j])
#     test_dict['left'].append(data_labels[i, -2])
#     test_dict['right'].append(data_labels[i, -1])
#
# print('Start saving data to csv file ...')
# dataframe_train = pd.DataFrame(train_dict)
# dataframe_train.to_csv('./datasets/train_data.csv', index=False)
#
# dataframe_test = pd.DataFrame(test_dict)
# dataframe_test.to_csv('./datasets/test_data.csv', index=False)