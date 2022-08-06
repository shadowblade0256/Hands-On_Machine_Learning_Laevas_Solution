from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_moons():
    X, y = make_moons(n_samples=500,noise=0.10)
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    return X_train, X_val, y_train, y_val, X, y

def poly_svm_fit(model,X,y):
    poly_svm_preprocess = Pipeline([
        ("poly_features",PolynomialFeatures(degree=4,include_bias=False)),
        ("scaler",StandardScaler())
    ])
    X = poly_svm_preprocess.fit_transform(X)
    model.fit(X,y)
    return model

def plot_decision_boundary(model, axis):
    poly = Pipeline([
        ("poly_features",PolynomialFeatures(degree=4,include_bias=False)),
        ("scaler",StandardScaler())
    ])
    x0,x1=np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1]-axis[0])*100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3]-axis[2])*100)).reshape(-1, 1)
    )
    X_new = poly.fit_transform(np.c_[x0.ravel(), x1.ravel()])                 #c_[]将两个数组以列的形式拼接起来，形成矩阵。
    y_predict = model.predict(X_new)

    zz = y_predict.reshape(x0.shape)                      #通过训练好的模型，预测平面上这些点的分类

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A','#FFF59D','#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

if __name__=="__main__":
    X_train, X_val, y_train, y_val, X, y = get_moons()
    svm_clf = SVC(degree=1,kernel="linear",C=10)
    svm_clf = poly_svm_fit(svm_clf,X_train,y_train)

    from matplotlib.colors import ListedColormap
    plot_decision_boundary(svm_clf,[X[:,0].min(),X[:,0].max(),X[:,1].min(),X[:,1].max()])
    plt.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm_r")
    plt.show()