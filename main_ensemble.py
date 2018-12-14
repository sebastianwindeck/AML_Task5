import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from helpers.io import outputter, inputter_train, inputter_test
from helpers.preprocessing import movingaverage
from helpers.plotter import plot_confusion_matrix
from joblib import load
from mlxtend.classifier import EnsembleVoteClassifier

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

clf1 = load(os.getcwd() + '/model/gnaive.joblib')
clf2 = load(os.getcwd() + '/model/rfc500.joblib')

evc = EnsembleVoteClassifier(clfs=[clf1, clf2], weights=[2,1], refit=False, voting='soft')

'''Fit model'''
evc.fit(X , y)
print('Fitted.')

y_pred = evc.predict(X_test)
print('Predicted.')
print('')
print('--- Confusion matrix ---')

classes = np.unique(y_test)
plot_confusion_matrix(classes,y_true=y_test, y_pred=y_pred)

'''Predict test set'''

eeg1_t,eeg2_t, emg_t = inputter_test()

data_t = np.concatenate((np.reshape(eeg1_t, (-1, 128)),
                         np.reshape(eeg2_t, (-1, 128)),
                         np.reshape(emg_t, (-1, 128))), axis=1)

del eeg1_t
del eeg2_t
del emg_t
print("Data format: ", data_t.shape)

y_pred_t = evc.transform(data_t)
y_pred_t = np.reshape(y_pred_t, (-1, 4))
# TODO: evtl modus plus prior if even
y_pred_t = np.mean(y_pred_t, axis=1)


smoothened = np.round(movingaverage(y_pred_t, 3))
print(smoothened.shape)
smoothened = np.add(smoothened, 1)
plt.plot(np.add(y_pred_t, 1), alpha=0.15)
plt.plot(smoothened[:])
plt.show()


outputter(smoothened)
