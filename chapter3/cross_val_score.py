from sgd_5 import sgd_classify
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def cross_validation(model,X,y):
    cross_vs = cross_val_score(model,X,y,cv=3,scoring="accuracy")
    print("Cross validation (Accuracy):")
    print(cross_vs)
    print("Mean: ",np.average(cross_vs))
    print("Standard deviation: ",np.std(cross_vs))
    print("")
    cross_vp = cross_val_predict(model,X,y,cv=3)
    print(confusion_matrix(y,cross_vp))

if __name__=='__main__':
    train = pd.read_csv("datasets\\mnist_train.csv").drop(labels=["Unnamed: 0"],axis=1)
    test = pd.read_csv("datasets\\mnist_test.csv").drop(labels=["Unnamed: 0"],axis=1)
    train_X, train_y = np.asarray(train.drop(labels=["label"],axis=1)), np.asarray(train['label'])
    test_X, test_y = np.asarray(test.drop(labels=["label"],axis=1)), np.asarray(test['label'])
    train_y_5 = (train_y == 5)
    sgd_clf = sgd_classify(train_X, train_y_5)
    cross_validation(sgd_clf, train_X, train_y_5)
