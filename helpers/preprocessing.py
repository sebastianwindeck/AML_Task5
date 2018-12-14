import numpy as np
import matplotlib.pyplot as plt


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'same')
    return sma


def transform_proba(y_pred, exponential=True):
    # TODO: evtl modus plus prior if even

    print(y_pred.shape)
    # Use exponential of the probabilities
    if exponential:
        y_pred = np.expm1(y_pred)
    # Sum over estimators
    for i in range(y_pred.shape[0]):
        plt.figure()
        plt.plot(y_pred[0][:500])
        plt.title('Sample: {}'.format(i))
        plt.show()
    y_pred = np.sum(y_pred, axis=0)
    # Reshape to 4 steps in epoch
    y_pred = np.reshape(y_pred, (-1, 4, 3))
    # Sum over epoch
    y_pred = np.sum(y_pred, axis=1)
    # Get the best class
    y_pred = np.argmax(y_pred, axis=1)
    # Transform to get class labels
    y_pred = np.add(y_pred, 1)
    print('Unique values: ', np.unique(y_pred))
    print(y_pred.shape)

    return y_pred
