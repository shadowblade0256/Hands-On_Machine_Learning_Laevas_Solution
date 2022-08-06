from operator import ge
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_dataset():
    X,y = load_iris(return_X_y=True)
    y_class = (y==2).astype(np.int32)
    return X,y_class

# 辅助：绘制决策边界
# axis指定坐标的范围，格式为[xmin,xmax,ymin,ymax]
def plot_decision_boundary(model, axis):
    # 边界中每个刻度间插值200个点
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int(
            (axis[1]-axis[0])*200)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int(
            (axis[3]-axis[2])*200)).reshape(-1, 1)
    )
    # c_[]将两个数组以列的形式拼接起来，形成矩阵。
    xx = np.c_[x0.ravel(),x1.ravel()]
    y_predict = model.predict(xx)

    zz = y_predict.reshape(x0.shape)  # 通过训练好的模型，预测平面上这些点的分类

    # 使用matplotlib.pyplot中等高线图(contour、contourf)绘制决策边界
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#EEEEEE', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidths=2, linestyles='dashed', cmap=custom_cmap, alpha=0.5)
    plt.contour(x0, x1, zz, linewidths=2, linestyles='solid', colors="#000000")
    # plt.plot(xx[:,0],xx[:,1])

if __name__=='__main__':
    X, y = get_dataset()
    svc_soft = SVC(kernel="linear",C=1)
    svc_soft.fit(X[:,[0,2]],y)
    plot_decision_boundary(svc_soft,[X[:,0].min()-1,X[:,0].max()+1,X[:,2].min()-1,X[:,2].max()+1])
    plt.scatter(X[:,0],X[:,2],c=y,cmap="coolwarm_r")
    plt.show()