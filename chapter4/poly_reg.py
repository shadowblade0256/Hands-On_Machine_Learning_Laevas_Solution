from os import X_OK
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(m=1000):
    X = np.random.rand(m,1)
    y = 4.5*(X**2)+2.5*X+3+np.random.rand(m,1)
    return X, y

def add_poly_features(X):
    poly_transformer = PolynomialFeatures(degree=2,include_bias=False)
    X_new = poly_transformer.fit_transform(X)
    return X_new

def visualize(X,y,y_baseline=None):
    plt.scatter(X,y,label="Prediction")
    if y_baseline is not None:
        plt.scatter(X,y_baseline,label="True")
    plt.xlabel("GDP per capita")
    plt.ylabel("Life satisfaction")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X, y = generate_data()
    X_bi = add_poly_features(X)
    bi_reg = LinearRegression()
    bi_reg.fit(X_bi,y)
    print(bi_reg.intercept_,bi_reg.coef_)
    y_pred = bi_reg.predict(X_bi)
    visualize(X,y_pred,y_baseline=y)
