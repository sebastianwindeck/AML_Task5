import numpy as np
import os

train_eeg1 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_eeg2 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_emg = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_labels = np.genfromtxt('data/import/train_labels.csv', delimiter=',', skip_header=1)
print("Imported everything.")

data_shape = (64800, 4, 128)

train_eeg1 = np.reshape(train_eeg1[:,1:], data_shape)
train_eeg2 = np.reshape(train_eeg2[:,1:], data_shape)
train_emg = np.reshape(train_emg[:,1:], data_shape)
train_labels = train_labels[:,1:]
print("Reshape it.")

train_eeg1 = np.reshape(np.mean(train_eeg1,axis=1),(3,-1,128))
print("Train shape: ", train_eeg1.shape)
train_eeg2 = np.reshape(np.mean(train_eeg2,axis=1),(3,-1,128))
train_emg = np.reshape(np.mean(train_emg,axis=1),(3,-1,128))
train_labels = np.reshape(train_labels, (3,-1))
print("Reshape it.")
print("Mean it.")

path = os.getcwd()
np.save(path + '/data/numpy/data_eeg1.npy', train_eeg1)
np.save(path + '/data/numpy/data_eeg2.npy', train_eeg2)
np.save(path + '/data/numpy/data_emg.npy', train_emg)

np.save(path + '/data/numpy/labels.npy', train_labels)

print('Exported everything')


