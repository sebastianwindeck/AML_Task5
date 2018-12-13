from sklearn.naive_bayes import GaussianNB
import numpy as np
def mnb (X,Y, test):
    y_pred = np.zeros((4,Y.shape[0],))
    y_pred_total = np.zeros((Y.shape[0],))
    for i in range(X.shape[1]):
        mnb = GaussianNB()
        mnb.fit(X[:,i , :], np.ravel(Y))
        #y_test = mnb.predict(test)
        y_pred[i, :, :] = mnb.predict_proba(X[:, i, :])

     y_pred_total = np.sum(y_pred, axis=0)
     y_pred_total = np.amax(y_pred,axis=1)

    return: y_pred_total


