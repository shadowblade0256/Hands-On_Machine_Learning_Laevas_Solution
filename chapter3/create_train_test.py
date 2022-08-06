from fetch_mnist import get_mnist
import numpy as np
import pandas as pd

def split_train_test(X,y,threshold=60000):
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    train_output = pd.DataFrame(data=np.c_[X_train,y_train],
        columns=["features_"+str(i) for i in range(X_train.shape[1])]+['label'],
        index=None)
    test_output = pd.DataFrame(data=np.c_[X_test,y_test],
        columns=["features_"+str(i) for i in range(X_test.shape[1])]+['label'],
        index=None)
    train_output.to_csv("datasets/mnist_train.csv")
    test_output.to_csv("datasets/mnist_test.csv")
    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    X, y = get_mnist()
    split_train_test(X, y)