import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Lambda, Concatenate
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import EarlyStopping
from numpy.core.multiarray import ndarray
from scipy.signal import savgol_filter

from helpers.io import inputter_train, inputter_test, outputter
from helpers.preprocessing import transform_proba

stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1)


def build(_base_shape):
    inputer = Input(shape=_base_shape, name='input')
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(inputer)

    conv1 = Conv1D(filters=16, kernel_size=11, activation='relu', padding='valid', name='conv1')(split[0])
    maxpool1 = MaxPooling1D()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='valid', name='conv2')(maxpool1)
    maxpool2 = MaxPooling1D()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='valid', name='conv3')(maxpool2)
    maxpool3 = MaxPooling1D()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='valid', name='conv4')(maxpool3)
    maxpool4_1 = MaxPooling1D()(conv4)

    conv1 = Conv1D(filters=16, kernel_size=11, activation='relu', padding='valid', name='conv1_2')(split[1])
    maxpool1 = MaxPooling1D()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='valid', name='conv2_2')(maxpool1)
    maxpool2 = MaxPooling1D()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='valid', name='conv3_2')(maxpool2)
    maxpool3 = MaxPooling1D()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='valid', name='conv4_2')(maxpool3)
    maxpool4_2 = MaxPooling1D()(conv4)

    conv1 = Conv1D(filters=16, kernel_size=11, activation='relu', padding='valid', name='conv1_3')(split[1])
    maxpool1 = MaxPooling1D()(conv1)
    conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='valid', name='conv2_3')(maxpool1)
    maxpool2 = MaxPooling1D()(conv2)
    conv3 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='valid', name='conv3_3')(maxpool2)
    maxpool3 = MaxPooling1D()(conv3)
    conv4 = Conv1D(filters=128, kernel_size=5, activation='relu', padding='valid', name='conv4_3')(maxpool3)
    maxpool4_3 = MaxPooling1D()(conv4)

    merger = Concatenate(axis=1)([maxpool4_1, maxpool4_2, maxpool4_3])

    flatten = Flatten()(merger)
    dense1 = Dense(1024, activation='relu', name='dense1')(flatten)
    dense2 = Dense(512, activation='relu', name='dense2')(dense1)
    outputer = Dense(3, activation='softmax')(dense2)

    _model = Model(inputs=inputer, outputs=outputer)  # type: Model
    return _model


eeg1, eeg2, emg, lab = inputter_train()

print('Each data input shape: ', eeg1.shape)
data = np.concatenate((np.reshape(eeg1, (-1, 128)), np.reshape(eeg2, (-1, 128)), np.reshape(emg, (-1, 128))), axis=1)
data = data[..., np.newaxis]
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
labels = to_categorical(labels, num_classes=None)  # type: ndarray

base_shape = (data.shape[1], data.shape[2])
print('Input shape: ', base_shape)
print('Label shape: ', labels.shape)
print('Input done.')

model = build(base_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

print(model.summary())
plot_model(model, to_file=os.getcwd() + '/data/' + str(time.strftime("%Y%m%d-%H%M%S")) + '_model.png', show_shapes=True,
           show_layer_names=True, rankdir='TB')

print("Unique labels: ", np.unique(lab))
model.fit(data, labels, batch_size=128, epochs=50, verbose=1, validation_split=0.1,
          callbacks=[stopper])
model.save_weights("/model/conv2d_model.h5")


eeg1_t, eeg2_t, emg_t = inputter_test()

data_t = np.concatenate((np.reshape(eeg1_t, (-1, 128)),
                         np.reshape(eeg2_t, (-1, 128)),
                         np.reshape(emg_t, (-1, 128))), axis=1)

data_t = data_t[..., np.newaxis]

del eeg1_t
del eeg2_t
del emg_t
print("Data format: ", data_t.shape)

y_pred_t = model.predict(data_t)
y_pred_t = transform_proba(y_pred=y_pred_t, exponential=False)

smoothened = np.reshape(y_pred_t, (2, -1))
smoothened = np.round(smoothened)
print(smoothened.shape)
smoothened = savgol_filter(smoothened, polyorder=1, axis=1, window_length=5, mode='nearest')
plt.plot(smoothened.T[:5000, 1])
smoothened = np.round(smoothened)
print(smoothened.shape)
# plt.plot(y_pred_t, alpha=0.15)
plt.plot(smoothened.T[:5000, 1])
plt.show()
smoothened = np.reshape(smoothened, (-1, 1))

outputter(smoothened)
