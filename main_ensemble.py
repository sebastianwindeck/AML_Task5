import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from helpers.io import outputter
from helpers.preprocessing import movingaverage

eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

'''Reshape data set'''

print(eeg1.shape)
data = np.concatenate((np.reshape(eeg1, (-1, 128)), np.reshape(eeg2, (-1, 128)), np.reshape(emg, (-1, 128))), axis=1)
print("Data format: ", data.shape)

print(lab.shape)
labels = np.reshape(lab, (-1, 1))
labels = np.concatenate((labels, labels, labels, labels), axis=1)
print(labels.shape)
labels = np.reshape(labels, (-1, 1))
labels = np.subtract(labels, 1)

print("Label format: ", labels.shape)

'''Split data set'''

X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
y = np.ravel(y)

'''Create model'''

clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                          random_state=1, verbose=10)
clf2 = RandomForestClassifier(n_estimators=200, random_state=1, verbose=10)
clf3 = GaussianNB()
clf4 = SVC(kernel='linear', class_weight='balanced', verbose=10, probability=True)
clf5 = SVC(kernel='rbf', class_weight='balanced', verbose=10, probability=True)

eclf1 = VotingClassifier(estimators=[
    ('lr', clf1), ('rf', clf2), ('gnb', clf3), ('l_svm', clf4), ('r_svm', clf5)], voting='hard', n_jobs=-1)

'''Fit model'''

eclf1 = eclf1.fit(X, y)
print('Fitted.')

y_pred = eclf1.predict(X_test)
print('Predicted.')
confusion_matrix(y, y_pred)

del eeg1
del eeg2
del emg

'''Predict test set'''

eeg1_t = np.load(os.getcwd() + '/data/numpy/test_eeg1.npy')
eeg2_t = np.load(os.getcwd() + '/data/numpy/test_eeg2.npy')
emg_t = np.load(os.getcwd() + '/data/numpy/test_emg.npy')

data_t = np.concatenate((np.reshape(eeg1_t, (-1, 128)), np.reshape(eeg2_t, (-1, 128)), np.reshape(emg_t, (-1, 128))),
                        axis=1)
print("Data format: ", data_t.shape)

y_pred_t = eclf1.predict(data_t)
y_pred_t = np.reshape(y_pred_t, (-1, 4))
y_pred_t = np.mean(y_pred_t, axis=1)


smoothened = np.round(movingaverage(y_pred_t, 2))
print(smoothened.shape)
smoothened = np.add(smoothened, 1)
plt.plot(np.add(y_pred_t, 1), alpha=0.15)
plt.plot(smoothened[:])
plt.show()

y_pred_t = eclf1.predict(data_t)
y_pred_t = np.reshape(y_pred_t, (-1, 4))
y_pred_t = np.mean(y_pred_t, axis=1)

del eeg1_t
del eeg2_t
del emg_t

outputter(smoothened)
