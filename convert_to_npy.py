import numpy as np
import os

train_eeg1 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_eeg2 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_emg = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_labels = np.genfromtxt('data/import/train_labels.csv', delimiter=',', skip_header=1)
print("Imported everything.")

data_shape = (64800, 4, 128)

train_eeg1 = np.reshape(train_eeg1,(3,-1,4,128))
print("Train shape: ", train_eeg1.shape)
train_eeg2 = np.reshape(train_eeg2,(3,-1,4,128))
train_emg = np.reshape(train_eeg2,(3,-1,4,128))
train_labels = np.reshape(train_labels, (3,-1))
print("Reshape it.")
print("Mean it.")

path = os.getcwd()
np.save(path + '/data/numpy/data_eeg1.npy', train_eeg1)
np.save(path + '/data/numpy/data_eeg2.npy', train_eeg2)
np.save(path + '/data/numpy/data_emg.npy', train_emg)

np.save(path + '/data/numpy/labels.npy', train_labels)

print('Exported everything')


