from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_moons():
    X, y = make_moons(n_samples=1000,noise=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    return X_train, X_val, y_train, y_val, X, y

def poly_svm_fit(model,X,y):
    poly_svm_preprocess = Pipeline([
        ("scaler",StandardScaler())
    ])
    X = poly_svm_preprocess.fit_transform(X)
    model.fit(X,y)
    return model

def plot_decision_boundary(model, axis):
    poly = Pipeline([
        ("scaler",StandardScaler())
    ])
    x0,x1=np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*200)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*200)).reshape(-1, 1)
    )
    X_new = poly.fit_transform(np.c_[x0.ravel(), x1.ravel()])                 #c_[]将两个数组以列的形式拼接起来，形成矩阵。
    y_predict = model.predict(X_new)

    zz = y_predict.reshape(x0.shape)                      #通过训练好的模型，预测平面上这些点的分类

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#EEEEEE','#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.contour(x0, x1, zz, linewidths=1.5, colors="#8B8B8B", linestyles="dashed")

def plot_learning_curve(model,X_train,X_val,y_train,y_val):
    poly = Pipeline([
        ("scaler",StandardScaler())
    ])
    train_errors, val_errors = [], []
    # Run again if raised ValueError
    # This may be caused by an improper split of X_train / X_val
    for m in range(3,len(X_train)):
        X_batch = poly.fit_transform(X_train[:m])
        model.fit(X_batch,y_train[:m])
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_errors.append(f1_score(y_train,y_train_pred))
        val_errors.append(f1_score(y_val,y_val_pred))
    print("Train f1 score =",train_errors[-1])
    print("Validate f1 score =",val_errors[-1])
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="Train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=2,label="Validation")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X_train, X_val, y_train, y_val, X, y = get_moons()
    svm_clf = SVC(kernel="rbf",gamma=1,C=0.5)
    svm_clf = poly_svm_fit(svm_clf,X_train,y_train)

    plot_decision_boundary(svm_clf,[X[:,0].min(),X[:,0].max(),X[:,1].min(),X[:,1].max()])
    plt.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm_r")
    plt.show()

    plot_learning_curve(svm_clf,X_train,X_val,y_train,y_val)