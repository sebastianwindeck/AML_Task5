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
from helpers.plotter import plot_confusion_matrix
from helpers.preprocessing import movingaverage
from joblib import dump
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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
X = data
y = labels

'''Create model'''

clf = SVC(kernel='rbf', C=1, gamma='auto')

'''Feature selection'''
f_select = SelectKBest(f_classif, k=200)

'''pipe'''
pipeline = Pipeline([
    ('f_select', f_select),
    ('SVC', clf)
])

'''parameters for Gsearch'''
parameters = {
    'f_select__k': [150, 250],
    # 'SVC__degree' : [2,5,8],
    # höher -> 1000
    'SVC__C': [0.1, 1, 10]
}

'''Gridsearch'''
CV = GridSearchCV(pipeline, parameters, scoring='balanced_accuracy', n_jobs=-1, cv=8, verbose=2)

'''Fit model'''
clf = CV.fit(X, np.ravel(y))

dump(clf, os.getcwd() + '/model/rbf1.joblib')
print('Fitted.')

y_pred = clf.predict(X)
print(y_pred)
print('Predicted.')
print('')
print('--- Confusion matrix ---')
print(np.ravel(y))
print(y_pred)

classes = np.unique(y)
plot_confusion_matrix(classes, y_true=y, y_pred=y_pred)

del eeg1
del eeg2
del emg

'''Predict test set'''

eeg1_t = np.load(os.getcwd() + '/data/numpy/test_eeg1.npy')
eeg2_t = np.load(os.getcwd() + '/data/numpy/test_eeg2.npy')
emg_t = np.load(os.getcwd() + '/data/numpy/test_emg.npy')

data_t = np.concatenate((np.reshape(eeg1_t, (-1, 128)),
                         np.reshape(eeg2_t, (-1, 128)),
                         np.reshape(emg_t, (-1, 128))),
                        axis=1)
del eeg1_t
del eeg2_t
del emg_t

print("Data format: ", data_t.shape)

y_pred_t = clf.predict(data_t)
y_pred_t = np.reshape(y_pred_t, (-1, 4))
y_pred_t = np.mean(y_pred_t, axis=1)

smoothened = np.round(movingaverage(y_pred_t, 2))
print(smoothened.shape)
smoothened = np.add(smoothened, 1)
plt.plot(np.add(y_pred_t, 1), alpha=0.15)
plt.plot(smoothened[:])
plt.show()

outputter(smoothened)
