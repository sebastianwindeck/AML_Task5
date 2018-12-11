import numpy as np
import os

train_eeg1 = np.genfromtxt('data/import/train_eeg1.csv', delimiter=',', skip_header=1)
train_eeg2 = np.genfromtxt('data/import/train_eeg2.csv', delimiter=',', skip_header=1)
train_emg = np.genfromtxt('data/import/train_emg.csv', delimiter=',', skip_header=1)
train_labels = np.genfromtxt('data/import/train_labels.csv', delimiter=',', skip_header=1)
print("Imported everything for train.")

train_eeg1 = train_eeg1[:, 1:]
train_eeg2 = train_eeg2[:, 1:]
train_emg = train_emg[:, 1:]
train_labels = train_labels[:, 1:]

train_eeg1 = np.reshape(train_eeg1, (3, -1, 4, 128))
print("Train shape: ", train_eeg1.shape)
train_eeg2 = np.reshape(train_eeg2, (3, -1, 4, 128))
train_emg = np.reshape(train_emg, (3, -1, 4, 128))
train_labels = np.reshape(train_labels, (3, -1))
print("Reshape it.")

path = os.getcwd()
np.save(path + '/data/numpy/data_eeg1.npy', train_eeg1)
np.save(path + '/data/numpy/data_eeg2.npy', train_eeg2)
np.save(path + '/data/numpy/data_emg.npy', train_emg)

np.save(path + '/data/numpy/labels.npy', train_labels)

print('Exported everything for train')

del train_labels
del train_eeg1
del train_eeg2
del train_emg


test_eeg1 = np.genfromtxt('data/import/test_eeg1.csv', delimiter=',', skip_header=1)
test_eeg2 = np.genfromtxt('data/import/test_eeg2.csv', delimiter=',', skip_header=1)
test_emg = np.genfromtxt('data/import/test_emg.csv', delimiter=',', skip_header=1)

print("Imported everything for test.")

test_eeg1 = test_eeg1[:, 1:]
test_eeg2 = test_eeg2[:, 1:]
test_emg = test_emg[:, 1:]

test_eeg1 = np.reshape(test_eeg1, (2, -1, 4, 128))
print("Test shape: ", test_eeg1.shape)
test_eeg2 = np.reshape(test_eeg2, (2, -1, 4, 128))
test_emg = np.reshape(test_emg, (2, -1, 4, 128))
print("Reshape it.")

path = os.getcwd()
np.save(path + '/data/numpy/test_eeg1.npy', test_eeg1)
np.save(path + '/data/numpy/test_eeg2.npy', test_eeg2)
np.save(path + '/data/numpy/test_emg.npy', test_emg)

print('Exported everything for test')

del test_eeg1
del test_eeg2
del test_emg