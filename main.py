import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from keras.layers import RNN, LSTM
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Concatenate

eeg1 = np.load(os.getcwd() + '/data/numpy/data_eeg1.npy')
eeg2 = np.load(os.getcwd() + '/data/numpy/data_eeg2.npy')
emg = np.load(os.getcwd() + '/data/numpy/data_emg.npy')
lab = np.load(os.getcwd() + '/data/numpy/labels.npy')


shaper = eeg1.shape
print('Shape: ', shaper)
print('Input done.')


def build(shaper):
    eeg1_inputer = Input(shape=shaper, name='eeg1_input')
    eeg2_inputer = Input(shape=shaper, name='eeg2_input')
    emg_inputer = Input(shape=shaper, name='emg_input')

    lstm_out1 = LSTM(32)(eeg1_inputer)
    lstm_out2 = LSTM(32)(eeg2_inputer)
    lstm_out3 = LSTM(32)(emg_inputer)


    merged = Concatenate([lstm_out1, lstm_out2, lstm_out3])
    outputer = Dense(21600, activation='tanh')(merged)
    



    model = Model(inputs=[eeg1_inputer,eeg2_inputer,emg_inputer], outputs=outputer)
    return model

model = build(shaper)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

print(model.summary())


labels = np.subtract(labels, 1)
print("Unique labels: ", np.unique(labels) )
model.fit(data, labels, batch_size=16, epochs=10, verbose=1, callbacks=None, validation_split=0.2)

y_pred = model.predict(sub1)
print(y_pred)
y_pred = np.add(np.argmax(y_pred, axis=1) ,1)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cnf_matrix = confusion_matrix(lab1, y_pred)
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
