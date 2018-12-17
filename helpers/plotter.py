import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    eps = 10**(-6)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+eps)
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

    plt.show()

def plot_confusion_matrix(classes, y_true, y_pred):

    cnf_matrix = confusion_matrix(y_true=np.ravel(y_true), y_pred=y_pred)
    np.set_printoptions(precision=2)

    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    _plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')


def plotter_input(data, labels):

    nr = 3
    nc = 3
    fig = plt.figure()
    j = 0
    for i in (0, 1, 2):
        ind = np.where(labels == i)
        ind = ind[0]
        print(ind.shape)
        for k in (0, 1, 2):
            ctr = j + 1
            print(ctr)
            ax = fig.add_subplot(nr, nc, ctr)
            data_p = data[ind[0], k * 128:(k + 1) * 128]
            ax.plot(data_p, color='b', alpha=0.2)
            data_p = data[ind[1], k * 128:(k + 1) * 128]
            ax.plot(data_p, color='b', alpha=0.2)
            data_p = data[ind[2], k * 128:(k + 1) * 128]
            ax.plot(data_p, color='b', alpha=0.2)
            ax.set_title(label=('Class: ', i, ' Type: ', k))
            j = j + 1

    plt.show()