import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, Dense
from keras.layers import concatenate
from keras.utils import plot_model, to_categorical

from helpers.plotter import plot_confusion_matrix


def build(base_shape):
    eeg1_inputer = Input(shape=base_shape, name='eeg1_input')
    eeg2_inputer = Input(shape=base_shape, name='eeg2_input')
    emg_inputer = Input(shape=base_shape, name='emg_input')

    lstm_out1 = LSTM(64, return_sequences=True, recurrent_activation='hard_sigmoid', name='lstm_eeg1')(eeg1_inputer)
    lstm_out2 = LSTM(64, return_sequences=True, recurrent_activation='hard_sigmoid', name='lstm_eeg2')(eeg2_inputer)
    lstm_out3 = LSTM(64, return_sequences=True, recurrent_activation='hard_sigmoid', name='lstm_emg')(emg_inputer)

    merged = concatenate([lstm_out1, lstm_out2, lstm_out3], axis=2, name='merger')

    lstm_merged = LSTM(3, activation='softmax', recurrent_activation='hard_sigmoid', name='lstm_merged', return_sequences=True)(merged)

    _model = Model(inputs=[eeg1_inputer, eeg2_inputer, emg_inputer], outputs=lstm_merged)  # type: Model
    return _model


eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

eeg1 = np.mean(eeg1, axis=2)
eeg2 = np.mean(eeg2, axis=2)
emg = np.mean(emg, axis=2)

lab = np.subtract(lab, 1)
lab = to_categorical(lab, num_classes=None)

base_shape = (eeg1.shape[1], eeg1.shape[2])
print('Input shape: ', base_shape)
print('Label shape: ', lab.shape)
print('Input done.')

model = build(base_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

print(model.summary())
plot_model(model, to_file=os.getcwd() + '/data/model.png', show_shapes=False, show_layer_names=True, rankdir='TB')

print("Unique labels: ", np.unique(lab))
model.fit({'eeg1_input': eeg1, 'eeg2_input': eeg2, 'emg_input': emg}, lab, batch_size=1, epochs=10, verbose=1,
          callbacks=None)

y_pred = model.predict({'eeg1_input': eeg1, 'eeg2_input': eeg2, 'emg_input': emg})

print(y_pred[1,1,:])
y_pred = np.argmax(y_pred, axis=2)
y_pred = np.add(y_pred, 1)


plt.plot(y_pred[0])
plt.plot(y_pred[1])
plt.plot(y_pred[2])
plt.show()
