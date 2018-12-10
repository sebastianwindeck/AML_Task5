import numpy as np
import os

train_eeg1 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_eeg2 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_emg = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
print("Imported everything.")

data_shape = (64800, 4, 128)

train_eeg1 = np.reshape(train_eeg1[:,1:], data_shape)
train_eeg2 = np.reshape(train_eeg2[:,1:], data_shape)
train_emg = np.reshape(train_emg[:,1:], data_shape)
print("Reshape it.")

train_eeg1 = train_eeg1[..., np.newaxis]
print(train_eeg1.shape)
train_eeg2 = train_eeg2[..., np.newaxis]
train_emg = train_emg[..., np.newaxis]
print("Add axis.")

train_data = np.concatenate((train_eeg1, train_eeg2, train_emg), axis=3)
train_data = np.split(train_data, 3, axis=0)
data_sub1 = train_data[0]
data_sub2 = train_data[1]
data_sub3 = train_data[2]
print("Split it.")

path = os.getcwd()
np.save(path + 'data/numpy/data_sub1.npy', data_sub1)
np.save(path + 'data/numpy/data_sub2.npy', data_sub2)
np.save(path + 'data/numpy/data_sub3.npy', data_sub3)

print('Exported everything')