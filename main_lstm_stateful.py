import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM
from keras.models import Model
from keras.layers import Input, Reshape, Flatten, Dense
from keras.layers import concatenate
from keras.utils import plot_model, to_categorical

from helpers.plotter import plot_confusion_matrix


def build(base_shape):
    inputer = Input(batch_shape=base_shape, name='input')
    lstm = LSTM(3, return_sequences=False, stateful=True, recurrent_activation='hard_sigmoid', name='lstm_state')(inputer)

    _model = Model(inputs=inputer, outputs=lstm)  # type: Model
    return _model


eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
print(eeg1.shape)
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')

eeg1 = eeg1[0]
eeg2 = eeg2[0]
emg = emg[0]
data = np.concatenate((eeg1,eeg2,emg),axis=2)
lab = lab[0]
lab = np.subtract(lab, 1)

encoded_lab= to_categorical(lab, num_classes=3)

base_shape = (1, data.shape[1], data.shape[2])
print('Input shape: ', base_shape)
print('Label shape: ', encoded_lab.shape)
print('Input done.')

model = build(base_shape)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

print(model.summary())
plot_model(model, to_file=os.getcwd() + '/data/' + str(time.strftime("%Y%m%d-%H%M%S")) + '_model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

print("Unique labels: ", np.unique(lab))
model.fit(data, encoded_lab, batch_size=1, epochs=100, verbose=1,
          callbacks=None)

y_pred = model.predict(data, batch_size=1, verbose=1)

print(y_pred[1,:])
y_pred = np.argmax(y_pred, axis=1)
y_pred = np.add(y_pred, 1)


plt.plot(y_pred)

plt.show()


cnf_matrix = confusion_matrix(np.reshape(lab,(21600,)), np.reshape(y_pred,(21600,)))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1,2],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1,2], normalize=True,
                      title='Normalized confusion matrix')

plt.show()