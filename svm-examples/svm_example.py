from sklearn.pipeline import Pipeline
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 线性SVM分类器
class CustomLinearSVC():
    def __init__(self, C=1, loss=None):
        self.C = C
        self.loss = None
        self.w = None
        self.b = None

    def fit(self,X,y):
        pass

# 载入数据集
def get_dataset():
    #X2, y2 = make_moons(n_samples=500, noise=0.07, random_state=1000)
    X2 = (np.random.uniform(low=-4.5,high=4.5,size=50)).T
    y2 = (3*X2**2-5>=10).astype(np.int32)
    y2[y2==0]=-1
    return X2, y2

# 辅助：绘制决策边界
def plot_decision_boundary(model, axis):
    scaler = Pipeline([
        ('scaler',StandardScaler())
    ])
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int(
            (axis[1]-axis[0])*200)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int(
            (axis[3]-axis[2])*200)).reshape(-1, 1)
    )
    # c_[]将两个数组以列的形式拼接起来，形成矩阵。
    if model.n_features_in_ == 2:
        xx = np.c_[x0.ravel(),x1.ravel()]
    else:
        xx = np.c_[x0.ravel()]
    # xx = scaler.fit_transform(xx)
    y_predict = model.predict(xx)

    zz = y_predict.reshape(x0.shape)  # 通过训练好的模型，预测平面上这些点的分类

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#EEEEEE', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidths=2, linestyles='dashed', cmap=custom_cmap, alpha=0.5)
    plt.contour(x0, x1, zz, linewidths=2, linestyles='dashed', colors="#9A9A9A")
    # plt.plot(xx[:,0],xx[:,1])

if __name__=="__main__":
    X, y = get_dataset()
    plt.scatter(X,[0.5 for _ in range(X.shape[0])],c=y,cmap="coolwarm_r")
    plt.show()
    X2 = X
    X = np.c_[X,X**2]
    # X = scaler.fit_transform(X)
    # X = np.c_[X[:,0],X[:,1]]
    svc_clf = SVC(kernel="linear",C=1e10)
    svc_clf.fit(X,y)
    plot_decision_boundary(svc_clf,[X[:,0].min()-1,X[:,0].max()+1,X[:,1].min()-1,X[:,1].max()+1])
    plt.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm_r")
    plt.show()
    svc_orig = SVC(kernel="poly",degree=2,C=1e10)
    svc_orig.fit(np.c_[X2],y)
    plot_decision_boundary(svc_orig,[X2.min()-1,X2.max()+1,0.4,0.6])
    plt.scatter(X2,[0.5 for _ in range(X.shape[0])],c=y,cmap="coolwarm_r")
    plt.show()
