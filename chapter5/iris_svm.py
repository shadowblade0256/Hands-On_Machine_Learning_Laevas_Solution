from sklearn import datasets
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_iris_set():
    iris = datasets.load_iris()
    X = iris["data"][:,(2,3)] # petal length, petal width
    y = (iris["target"]==2).astype(np.float64)
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3)
    return X_train, X_val, y_train, y_val

def linear_svc_with_fit(X,y):
    svm_clf = Pipeline([
        ("scaler",StandardScaler()),
        ("linear_svc",LinearSVC(C=1,loss="hinge")),
    ])
    svm_clf.fit(X,y)
    return svm_clf

def plot_learning_curve(model,X_train,X_val,y_train,y_val):
    train_errors, val_errors = [], []
    for m in range(5,len(X_train)):
        model.fit(X_train[:m],y_train[:m])
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_errors.append(f1_score(y_train,y_train_pred))
        val_errors.append(f1_score(y_val,y_val_pred))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="Train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=2,label="Validation")
    plt.legend()
    plt.show()
    
if __name__=="__main__":
    X_train, X_val, y_train, y_val = load_iris_set()
    svm_clf = LinearSVC()
    plot_learning_curve(svm_clf,X_train,X_val,y_train,y_val)

