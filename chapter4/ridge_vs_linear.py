from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(m=100):
    X = np.random.rand(m,1)
    y = -40*X*X*X*X-76.6*X*X*X+23.3*X*X+50.6*X+13.2+np.random.rand(m,1)
    return X, y

def add_poly_features(X):
    poly_transformer = PolynomialFeatures(degree=4,include_bias=False)
    X_new = poly_transformer.fit_transform(X)
    return X_new

def visualize(X,y,y_baseline1=None,y_baseline2=None):
    plt.scatter(X,y,label="Prediction")
    if y_baseline1 is not None:
        plt.scatter(X,y_baseline1,label="True")
    if y_baseline2 is not None:
        plt.scatter(X,y_baseline2,label="Comp")
    plt.legend()
    plt.show()

if __name__=='__main__':
    X, y = generate_data()
    X = add_poly_features(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    ridge_reg = Ridge(alpha=0,solver="cholesky")
    ridge_reg_2 = Ridge(alpha=1e-4,solver="cholesky")
    ridge_reg_3 = Ridge(alpha=1,solver="cholesky")
    ridge_reg.fit(X,y)
    ridge_reg_2.fit(X,y)
    ridge_reg_3.fit(X,y)
    y_pred_lin = lin_reg.predict(X)
    y_pred_ridge = ridge_reg.predict(X)
    y_pred_ridge_2 = ridge_reg_2.predict(X)
    y_pred_ridge_3 = ridge_reg_3.predict(X)
    # q3visualize(X[:,0],y_pred_lin,y_baseline1=y,y_baseline2=y_pred_ridge)
    visualize(X[:,0],y_pred_ridge,y_baseline1=y_pred_ridge_2,y_baseline2=y_pred_ridge_3)