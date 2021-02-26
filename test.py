# %%
import scipy.io as sio

# 读取数据文件
file_path = './datasets/data/ckm/'

# 读取原始数据
mat_raw = sio.loadmat(file_path + 'ckm1.mat')
raw_data = mat_raw['EEG']['data'][0, 0]
raw_data = raw_data * 1e-6  # 进行数据缩放

# 读取mark样本点
mat_pos = sio.loadmat(file_path + '位置ckm1.mat')
event_pos = mat_pos['sckm1'][:, 0]

# 读取mark标签
mat_lab = sio.loadmat(file_path + '标签ckm1.mat')

event_lab = mat_lab['yckm1'][:, 0]

# %%
import numpy as np

# 构建events数组
event_dur = np.ones(len(event_lab), dtype=int) * 4
events = np.column_stack((event_pos, event_dur, event_lab))

# %%
import mne

# 构建info对象
ch_names = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
    'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2',
    'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'EOG', 'AF7',
    'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5',
    'C1', 'C2', 'C6', 'CP3', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3',
    'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'Oz'
]
ch_types = ['eeg'] * 31 + ['eog'] + ['eeg'] * 32
s_rate = 1024  # Hz
info = mne.create_info(ch_names, ch_types=ch_types, sfreq=s_rate)
info.set_montage('standard_1005')  # 设置电极位置
# info.set_montage('standard_1020')  # 设置电极位置


# %%
# 构建raw对象
raw = mne.io.RawArray(raw_data, info)

print(raw.info['meas_date'])

# 构建注释annot
mapping = {1: 'left_hand', 2: 'right_hand'}
annot_from_event = mne.annotations_from_events(events=events,
                                               event_desc=mapping,
                                               sfreq=raw.info['sfreq'],
                                               orig_time=raw.info['meas_date'])


raw.set_annotations(annot_from_event)

# %%
# 选择通道
# yapf:disable
ch_picks = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP5', 'CP3', 'CP1',        'CP2', 'CP4', 'CP6'
]
# yapf:enable
raw_picks = raw.copy().pick_channels(ch_picks)

# %%
# 绘制原始脑电信号
import matplotlib.pyplot as plt

# raw_picks.plot()

# %%
# 绘制原始脑电信号功率谱
# fig = raw.plot_psd(tmax=np.inf, fmax=100)

# %%
# 重采样
raw.resample(sfreq=512)

# %%
# 带通滤波
fmin = 2.
fmax = 36.
raw_filter = raw.copy().filter(fmin, fmax)

# filter_params = mne.filter.create_picks(raw_downsampled.get_data(),
#                                          raw_downsampled.info['sfreq'],
#                                          l_freq=fmin,
#                                          h_freq=fmax)
# mne.viz.plot_picks(filter_params, raw.info['sfreq'], flim=(0.01, 100))
# plt.show()

# raw_filter.plot()
# plt.show()

# %%
# 构建epoch对象
event_from_annot, event_dict = mne.events_from_annotations(raw)
reject_criteria = dict(eeg=150e-6)  # 拒绝幅值超过150µV的epoch
epochs = mne.Epochs(raw,
                    event_from_annot,
                    tmin=-0.1,
                    tmax=3.9,
                    event_id=event_dict,
                    reject=reject_criteria,
                    preload=True)
print(epochs)

# 对epoch进行滤波
epochs.filter(fmin, fmax)
epochs.plot(events=event_from_annot, event_id=event_dict)
exit()

# 拒绝异常epoch

epochs.drop_bad(reject=reject_criteria)
print(epochs)

epochs.plot(events=event_from_annot, event_id=event_dict)
plt.show()

# %%
ch_s = [
    'Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
    'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7',
    'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'O2', 'AF7', 'AF3', 'AF4', 'AF8',
    'F5', 'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3',
    'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8',
    'TP7', 'TP8', 'PO7', 'PO8', 'Oz'
]
epochs_picks = epochs.pick_channels(ch_s)

epochs_picks.plot_sensors(kind='topomap', ch_type='all')
epochs_picks.plot_sensors(kind='3d', ch_type='all');

# %%
# epochs_picks['left_hand'].plot_psd_topomap()
# epochs_picks['right_hand'].plot_psd_topomap()

# %%
bands = [(8, 12, '8~12 Hz'), (12, 30, '18~30 Hz')]
epochs_picks['left_hand'].plot_psd_topomap(bands=bands,
                                           ch_type='eeg',
                                           normalize=True)
epochs_picks['right_hand'].plot_psd_topomap(bands=bands,
                                            ch_type='eeg',
                                            normalize=True);

# %%
# epochs_picks['left_hand'].plot_image(picks='eeg', combine='mean')
# epochs_picks['right_hand'].plot_image(picks='eeg', combine='mean')

# epochs_picks['left_hand'].plot_image(picks=['C3', 'C4'])
# epochs_picks['right_hand'].plot_image(picks=['C3', 'C4'])

# epochs_picks['left_hand'].plot_image(picks=['C3', 'C4'], combine='gfp')
# epochs_picks['right_hand'].plot_image(picks=['C3', 'C4'], combine='gfp')

# %%
# 绘制事件数组
fig = mne.viz.plot_events(event_from_annot,
                          sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp,
                          event_id=event_dict)

# %%
# 分别绘制左手想象和右手想象时C3、C4通道的功率谱
epochs['left_hand'].plot_psd(fmin=2,
                             fmax=35,
                             picks=['C3', 'C4'],
                             spatial_colors=True)
epochs['right_hand'].plot_psd(fmin=2,
                              fmax=35,
                              picks=['C3', 'C4'],
                              spatial_colors=True);

# %%
