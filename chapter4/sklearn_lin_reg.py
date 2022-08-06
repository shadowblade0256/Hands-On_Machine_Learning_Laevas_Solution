from sklearn.linear_model import LinearRegression
import matplotlib
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

if __name__=='__main__':
    X, y = data_load()
    visualize(X,y)
    lin_reg = LinearRegression()
    lin_reg.fit(X.reshape(-1,1),y)
    print(lin_reg.predict([[63544]]))
    print(lin_reg.coef_)
