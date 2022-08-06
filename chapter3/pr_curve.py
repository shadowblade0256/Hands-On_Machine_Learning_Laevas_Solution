from sgd_5 import sgd_classify
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np

def prt_curve_plot(model,X,y):
    y_pred = model.predict(X)
    print("Precision: ",precision_score(y,y_pred))
    print("Recall: ",recall_score(y,y_pred))
    model_score = cross_val_predict(model,X,y,cv=3,method="decision_function")
    precision, recall, threshold = precision_recall_curve(y,model_score)
    import matplotlib
    import matplotlib.pyplot as plt
    plt.plot(threshold,precision[:-1],"b--",label="Precision")
    plt.plot(threshold,recall[:-1],"g-",label="Recall")
    plt.xlabel("Threshold")
    plt.ylim([0,1])
    plt.legend(loc="upper left")
    plt.show()

def pr_curve_plot(model,X,y):
    y_pred = model.predict(X)
    print("Precision: ",precision_score(y,y_pred))
    print("Recall: ",recall_score(y,y_pred))
    model_score = cross_val_predict(model,X,y,cv=3,method="decision_function")
    precision, recall, threshold = precision_recall_curve(y,model_score)
    import matplotlib
    import matplotlib.pyplot as plt
    plt.plot(recall[:-1],precision[:-1],"r-")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

if __name__=='__main__':
    train = pd.read_csv("datasets\\mnist_train.csv").drop(labels=["Unnamed: 0"],axis=1)
    test = pd.read_csv("datasets\\mnist_test.csv").drop(labels=["Unnamed: 0"],axis=1)
    train_X, train_y = np.asarray(train.drop(labels=["label"],axis=1)), np.asarray(train['label'])
    test_X, test_y = np.asarray(test.drop(labels=["label"],axis=1)), np.asarray(test['label'])
    train_y_5 = (train_y == 5)
    sgd_clf = sgd_classify(train_X, train_y_5)
    # prt_curve_plot(sgd_clf, train_X, train_y_5)
    pr_curve_plot(sgd_clf,train_X,train_y_5)
