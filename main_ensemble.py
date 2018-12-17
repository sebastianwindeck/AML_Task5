import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from helpers.io import outputter, inputter_train, inputter_test
from helpers.preprocessing import transform_proba
from helpers.plotter import plot_confusion_matrix
from joblib import load
from mlxtend.classifier import EnsembleVoteClassifier
from scipy.signal import savgol_filter

eeg1, eeg2, emg, lab = inputter_train()

'''Reshape data set'''

data = np.concatenate((np.reshape(eeg1, (-1, 128)), np.reshape(eeg2, (-1, 128)), np.reshape(emg, (-1, 128))), axis=1)
print("Data format: ", data.shape)

del eeg1
del eeg2
del emg

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

'''Load model'''
'''Hier müssen alle relevanten Classifier geladen werden'''

clf1 = load(os.getcwd() + '/model/gnb.joblib')
clf2 = load(os.getcwd() + '/model/rfc750.joblib')
clf3 = load(os.getcwd() + '/model/ann.joblib')

evc = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 3, 4
                                                               ], refit=False, voting='soft')

'''Fit model'''
evc.fit(X, y)
print('Fitted.')

y_pred = evc.predict(X_test)
print('Predicted.')
print('')
print('--- Confusion matrix ---')

classes = np.unique(y_test)
plot_confusion_matrix(classes, y_true=y_test, y_pred=y_pred)

'''Predict test set'''

eeg1_t, eeg2_t, emg_t = inputter_test()

data_t = np.concatenate((np.reshape(eeg1_t, (-1, 128)),
                         np.reshape(eeg2_t, (-1, 128)),
                         np.reshape(emg_t, (-1, 128))), axis=1)

del eeg1_t
del eeg2_t
del emg_t
print("Data format: ", data_t.shape)

y_pred_t = evc.transform(data_t)
del data_t
y_pred_t = transform_proba(y_pred=y_pred_t, exponential=True)

smoothened = np.reshape(y_pred_t, (2, -1))
smoothened = np.round(smoothened)
print(smoothened.shape)
smoothened = savgol_filter(smoothened, polyorder=2, axis=1, window_length=9, mode='nearest')
plt.plot(smoothened.T[:5000, 1])
smoothened = np.round(smoothened)
print(smoothened.shape)
# plt.plot(y_pred_t, alpha=0.15)
plt.plot(smoothened.T[:5000, 1])
plt.show()
smoothened = np.reshape(smoothened, (-1, 1))

outputter(smoothened)
