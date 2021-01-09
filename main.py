import os
import numpy as np
import mne
import pandas as pd


plot_duration = 20
length = 80
task_names = ['Task1', 'Task2', 'Task3', 'Task4']
task_filenames = {'Task1': ['03', '07', '11'], 'Task2': ['04', '08', '12'],
                  'Task3': ['05', '09', '13'], 'Task4': ['06', '10', '14']}
# ch_names = ['C5..', 'C3..', 'C1..', 'C2..', 'C4..', 'C6..',
#             'Cp5.', 'Cp3.', 'Cp1.', 'Cp2.', 'Cp4.', 'Cp6.',
#             'P7..', 'P5..', 'P3..', 'P1..', 'P2..', 'P4..', 'P6..', 'P8..']
ch_names = ['C3..', 'C1..',  'Cp3.', 'Cp1.', 'P5..', 'P3..', 'P1..']
dataset_dir = './datasets/physionet_files'

# init 'data'
data = {}
for i in range(len(ch_names)):
    for j in range(length):
        data[ch_names[i] + '%d' % j] = []
for task_name in task_names:
    data[task_name] = []

mne.set_log_file('./mne_output', overwrite=True)

list_dirs = os.walk(dataset_dir)
for root, dirs, files in list_dirs:
    for filename in files:
        if filename.split('.')[0].split('R')[1] in ['01', '02']:
            continue
        raw = mne.io.read_raw_edf(os.path.join(root, filename))
        raw.load_data()
        raw = raw.pick_channels(ch_names)
        # print(filename + ':' + str(raw_data.shape))
        print(raw)
        # print(raw.annotations.duration)
        # print(raw.info)
        # raw.plot_psd(tmax=np.inf, fmax=64, average=True)
        raw_notch = raw.copy().notch_filter(freqs=60, method='spectrum_fit', filter_length='10s')
        # raw_notch.plot_psd(tmax=np.inf, fmax=64, average=True)

        cutoff = 0.25
        raw_highpass = raw_notch.copy().filter(l_freq=cutoff, h_freq=None)

        raw_data, times = raw_highpass[:]
        raw_data_len = raw_data.shape[1]

        start_index = 0
        while start_index + length <= raw_data_len:
            for i in range(len(ch_names)):
                ch_name = ch_names[i]
                for j in range(length):
                    data[ch_name + '%d' % j].append(raw_data[i][start_index + j])

            for task_name in task_names:
                if filename.split('.')[0].split('R')[1] in task_filenames[task_name]:
                    data[task_name].append(1)
                else:
                    data[task_name].append(0)
            start_index += length

print('Start generating csv ......')
dataframe = pd.DataFrame(data)
dataframe.to_csv('./data_%d.csv' % length, index=False)



# sample_data_folder = mne.datasets.sample.data_path('./datasets')
# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_filt-0-40_raw.fif')
# sample_data_raw_file = './datasets/physionet/S008/' + filename

# event, event_dict = mne.events_from_annotations(raw)
# print(event_dict)
# print(event)
#
# epochs = mne.Epochs(raw, event, event_id=event_dict)
# epochs.load_data()
# print(epochs)
# print(epochs.event_id)
# print(epochs.ch_names)
#
# epochs_fc = epochs.copy().pick(['Fc5.'])
# event_color = {
#     1: 'r',
#     2: 'g',
#     3: 'b',
# }
# epochs_fc.plot(n_epochs=3, events=event, event_color=event_color, event_id=event_dict)
