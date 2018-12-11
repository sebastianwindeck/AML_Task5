import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Dense, Concatenate

from helpers.plotter import plot_confusion_matrix

eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

shaper = eeg1.shape
print('Shape: ', shaper)
print('Input done.')


def build(base_shape):
    eeg1_inputer = Input(shape=base_shape, name='eeg1_input')
    eeg2_inputer = Input(shape=base_shape, name='eeg2_input')
    emg_inputer = Input(shape=base_shape, name='emg_input')

    lstm_out1 = LSTM(32)(eeg1_inputer)
    lstm_out2 = LSTM(32)(eeg2_inputer)
    lstm_out3 = LSTM(32)(emg_inputer)

    merged = Concatenate([lstm_out1, lstm_out2, lstm_out3])
    outputer = Dense(21600, activation='tanh', name='output')(merged)

    _model = Model(inputs=[eeg1_inputer, eeg2_inputer, emg_inputer], outputs=outputer)  # type: Model
    return _model


model = build(shaper)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

print(model.summary())

labels = np.subtract(lab, 1)
print("Unique labels: ", np.unique(labels))
model.fit({'eeg1_input': eeg1, 'eeg2_input': eeg2, 'emg_input': emg}, batch_size=1, epochs=1, verbose=1, callbacks=None)

y_pred = model.predict([eeg1, eeg2, emg])
print(y_pred)
y_pred = np.add(np.argmax(y_pred, axis=1), 1)

cnf_matrix = confusion_matrix(lab, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['0', '1', '2'], normalize=True,
                      title='Normalized confusion matrix')

plt.show()
