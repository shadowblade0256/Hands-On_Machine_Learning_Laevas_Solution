import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def data_load():
    X = pd.read_csv("datasets/lifesat/oecd_bli_2015.csv",thousands=",")
    y = pd.read_csv("datasets/lifesat/gdp_per_capita.csv",thousands=",",delimiter="\t",
        encoding="latin1",na_values="n/a")
    X_tot = X[X["INEQUALITY"]=="TOT"]
    X_tot = X_tot.pivot(index="Country",columns="Indicator",values="Value")
    full = X_tot.merge(y,left_on="Country",right_on="Country")
    raw_data = np.c_[full["2015"],full["Life satisfaction"]]
    return raw_data[:,0], raw_data[:,1]

def visualize(X,y):
    zipped = np.c_[X,y]
    zipped = np.sort(zipped,axis=0)
    plt.scatter(zipped[:,0],zipped[:,1])
    plt.xlabel("GDP per capita")
    plt.ylabel("Life satisfaction")
    plt.show()

def visual_comp(X,model1,model2,y):
    y = y.T[0]
    X = X.T[0]
    y2 = model2.coef_[0] * X + model2.intercept_[0]
    y1 = model1.thetas_[1,0] * X + model1.thetas_[0,0]
    #plt.scatter(X,y1,label="Custom LinReg")
    #plt.scatter(X,y2,label="Sklearn LinReg")
    plt.plot(X,y1,label="Custom LinReg")
    plt.plot(X,y2,label="Sklearn LinReg")
    plt.scatter(X,y,label="True")
    plt.xlabel("GDP per capita")
    plt.ylabel("Life satisfaction")
    plt.legend()
    plt.show()

class CustomLinearRegression():
    def __init__(self):
        self._theta = None
    def fit(self,X,y):
        X = np.c_[np.ones_like(X),X]
        self._theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)),X.T),y)
    def predict(self,X):
        Xb = np.c_[ np.ones_like(X),X].T
        y_ = np.matmul(self._theta.T,Xb)
        return y_
    @property
    def params_(self):
        return {'theta':self._theta}
    @property
    def thetas_(self):
        return self._theta

if __name__=='__main__':
    # X, y = data_load()
    X = 2*np.random.rand(50,1)
    y = 4*X*X+3+np.random.rand(50,1)
    lin_reg = CustomLinearRegression()
    lin_reg_comp = LinearRegression()
    lin_reg.fit(X,y)
    lin_reg_comp.fit(X,y)
    visual_comp(X,lin_reg,lin_reg_comp,y)
