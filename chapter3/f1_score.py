from sgd_5 import sgd_classify
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

def pr_score(model,X,y):
    y_pred = model.predict(X)
    print("Precision: ",precision_score(y,y_pred))
    print("Recall: ",recall_score(y,y_pred))
    print("F1: ",f1_score(y,y_pred))

if __name__=='__main__':
    train = pd.read_csv("datasets\\mnist_train.csv").drop(labels=["Unnamed: 0"],axis=1)
    test = pd.read_csv("datasets\\mnist_test.csv").drop(labels=["Unnamed: 0"],axis=1)
    train_X, train_y = np.asarray(train.drop(labels=["label"],axis=1)), np.asarray(train['label'])
    test_X, test_y = np.asarray(test.drop(labels=["label"],axis=1)), np.asarray(test['label'])
    train_y_5 = (train_y == 5)
    sgd_clf = sgd_classify(train_X, train_y_5)
    pr_score(sgd_clf, train_X, train_y_5)
