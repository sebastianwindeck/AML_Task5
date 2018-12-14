import os
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input
from keras.layers import LSTM, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from sklearn.metrics import confusion_matrix

from helpers.plotter import plot_confusion_matrix


def build(_base_shape):
    inputer = Input(shape=_base_shape, name='input')
    time_conv1d = TimeDistributed(Conv1D(filters=16, kernel_size=10, name='conv1d_1'))(inputer)
    maxpool = TimeDistributed(MaxPooling1D(name='max_pool_1'))(time_conv1d)
    time_conv1d = TimeDistributed(Conv1D(filters=32, kernel_size=10, name='conv1d_2'))(maxpool)
    maxpool = TimeDistributed(MaxPooling1D(name='max_pool_2'))(time_conv1d)
    flatten = TimeDistributed(Flatten(name='flatten'))(maxpool)
    dense = TimeDistributed(Dense(32, activation='relu', name='dense_1'))(flatten)
    lstm = LSTM(3, return_sequences=True, stateful=False, recurrent_activation='softmax',
               name='gru_1', activation='sigmoid')(dense)

    _model = Model(inputs=inputer, outputs=lstm)  # type: Model
    return _model


eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
print(eeg1.shape)
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
print(eeg2.shape)
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
print(emg.shape)
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

lab = np.subtract(lab, 1)
print("Unique labels: ", np.unique(lab))
encoded_lab = to_categorical(lab, num_classes=3)

eeg1 = np.mean(eeg1, axis=2)
eeg2 = np.mean(eeg2, axis=2)
emg = np.mean(emg, axis=2)

eeg1 = eeg1[..., np.newaxis]
eeg2 = eeg2[..., np.newaxis]
emg = emg[..., np.newaxis]

data = np.concatenate((eeg1, eeg2, emg), axis=3)

print("Data shape: ,", data.shape)

base_shape = (data.shape[1], data.shape[2], data.shape[3])
print('Input shape: ', base_shape)
print('Label shape: ', encoded_lab.shape)
print('Input done.')

model = build(base_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

print(model.summary())
plot_model(model, to_file=os.getcwd() + '/data/' + str(time.strftime("%Y%m%d-%H%M%S")) + '_model.png',
           show_shapes=True, show_layer_names=True, rankdir='TB')

model.fit(data, encoded_lab, batch_size=1, epochs=100, verbose=1,
          callbacks=None)

y_pred = model.predict(data)

print(y_pred[1, 1, :])
y_pred = np.argmax(y_pred, axis=2)
y_pred = np.add(y_pred, 1)

plt.plot(y_pred[0])
plt.plot(y_pred[1])
plt.plot(y_pred[2])
plt.show()