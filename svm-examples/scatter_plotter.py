import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_moons
from mpl_toolkits.mplot3d import Axes3D

if __name__=="__main__":
    # Linear-separatable dataset example
    # Use iris dataset
    # Features: Row #2 (Petal length) & Row #3 (Petal width)
    X,y = load_iris(return_X_y=True)
    y_class = (y==0).astype(np.int32)
    plt.scatter(X[:,2][y_class==0],X[:,3][y_class==0],cmap="coolwarm",label="Iris virginica")
    plt.scatter(X[:,2][y_class==1],X[:,3][y_class==1],cmap="coolwarm_r",label="Not iris virginica")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.legend()
    plt.show()

    # Linear-unseparatable dataset example
    # Use moon dataset
    # Features: Row #0 & Row #1
    X2,y2 = make_moons(n_samples=500,noise=0.06,random_state=875)
    plt.scatter(X2[:,0],X2[:,1],c=y2,cmap="coolwarm")
    plt.show()
    fig = plt.figure()
    ax3d = Axes3D(fig)
    XX, yy = X2[:,0], X2[:,1]
    zz = XX**3
    Xs = np.arange(-5,5,0.1)
    ys = np.arange(-5,5,0.1)
    Xs, ys = np.meshgrid(Xs, ys)
    zs = (-0.24853687-(-0.13427267*Xs)-(-3.47433188*ys))/0.41175165
    ax3d.scatter3D(XX,yy,zz,cmap="coolwarm")
    ax3d.plot_surface(Xs,ys,zs,rstride=1,cstride=1,cmap="jet",alpha=1)
    plt.show()

    # Dataset for soft margin example
    # Use iris dataset
    # Features: Row #2 (Petal length) & Row #0 (Sepal length)
    X3,y3 = load_iris(return_X_y=True)
    y_class_3 = (y3==2).astype(np.int32)
    plt.scatter(X3[:,0],X3[:,2],c=y_class_3,cmap="coolwarm")
    plt.show()

    # Scale-sensitive dataset example
    # Use transformed iris dataset
    X,y = load_iris(return_X_y=True)
    X[:,2] = X[:,2] * 0.1
    X[:,3] = X[:,3] * 50 
    y_class = (y==0).astype(np.int32)
    plt.scatter(X[:,2],X[:,3],c=y_class,cmap="coolwarm")
    plt.show()