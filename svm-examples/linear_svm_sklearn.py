from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 载入数据集
def load_iris():
    iris_frame = pd.read_csv("iris.csv")
    X = np.c_[iris_frame["petal length"],iris_frame["petal width"]]
    y = np.asarray(iris_frame["is iris virginica"])
    return X, y

def plot_scatter(X,y):
    # Linear-separatable dataset example
    # Use iris dataset
    # Features: Row #2 (Petal length) & Row #3 (Petal width)
    y_class = (y==0).astype(np.int32)
    plt.scatter(X[:,0][y_class==1],X[:,1][y_class==1],c="#7C7CFF",label="Iris virginica")
    plt.scatter(X[:,0][y_class==0],X[:,1][y_class==0],c="#FF7C7C",label="Not iris virginica")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend()

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
    plt.contour(x0, x1, zz, linewidths=2, linestyles='solid',colors="#000000")

if __name__=="__main__":
    X,y = load_iris()
    linear_svc = SVC(kernel="linear",C=1e10)
    linear_svc.fit(X,y)
    plot_decision_boundary(linear_svc,[-0.2,7.2,-0.1,2.7])
    plot_scatter(X,y)
    plt.show()
    