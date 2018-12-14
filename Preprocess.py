import numpy as np
import os

## train_eeg1
## train_eeg2
## train_emg
## train_lab

eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

eeg1_1 = eeg1[0, :, :, :]
eeg2_1 = eeg2[0, :, :, :]
emg_1 = emg[0, :, :, :]
y_1 = lab[0, :]
print(eeg1_1.shape)
print(eeg2_1.shape)
print(emg_1.shape)
print(y_1.shape)
