import mne

mne.set_log_file('./mne_output', overwrite=True)
filename = mne.datasets.eegbci.load_data(1, 4, path='./datasets/eegbci')[0]
raw = mne.io.read_raw_edf(filename, preload=True)

data = raw.get_data()
print(raw.annotations.onset)