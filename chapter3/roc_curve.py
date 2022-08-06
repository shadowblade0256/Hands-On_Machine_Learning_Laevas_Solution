from sgd_5 import sgd_classify
from dt_5 import dtree_classify
from rf_5 import rf_classify
from knn_5 import knn_classify
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import numpy as np

def roc_curve_plot(model,X,y,scoring_method):
    y_pred = model.predict(X)
    model_score_raw = cross_val_predict(model,X,y,cv=3,method=scoring_method)
    if scoring_method=='predict_proba':
        model_score = model_score_raw[:,1]
    else:
        model_score = model_score_raw
    false_pos, true_pos, threshold = roc_curve(y,model_score)
    import matplotlib
    import matplotlib.pyplot as plt
    plt.plot(false_pos,false_pos,"b--")
    plt.plot(false_pos,true_pos,"r-")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    print("ROC AUC score: ",roc_auc_score(y,model_score))

if __name__=='__main__':
    train = pd.read_csv("datasets\\mnist_train.csv").drop(labels=["Unnamed: 0"],axis=1)
    test = pd.read_csv("datasets\\mnist_test.csv").drop(labels=["Unnamed: 0"],axis=1)
    train_X, train_y = np.asarray(train.drop(labels=["label"],axis=1)), np.asarray(train['label'])
    test_X, test_y = np.asarray(test.drop(labels=["label"],axis=1)), np.asarray(test['label'])
    train_y_5 = (train_y == 5)
    sgd_clf = sgd_classify(train_X, train_y_5)
    dt_clf = dtree_classify(train_X,train_y_5)
    rf_clf = rf_classify(train_X,train_y_5)
    knn_clf = knn_classify(train_X,train_y_5)
    # prt_curve_plot(sgd_clf, train_X, train_y_5)
    # roc_curve_plot(sgd_clf,train_X,train_y_5,scoring_method="decision_function")
    roc_curve_plot(knn_clf,train_X,train_y_5,scoring_method="predict_proba")
    roc_curve_plot(rf_clf,train_X,train_y_5,scoring_method="predict_proba")
