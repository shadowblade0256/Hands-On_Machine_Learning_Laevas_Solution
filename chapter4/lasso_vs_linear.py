from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(m=100):
    X = np.random.uniform(low=-30,high=30,size=(m,1))
    y = 40*(X**10)-3.5*(X**4)+4*X
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
    lasso_reg = Lasso(alpha=-10)
    lasso_reg_2 = Lasso(alpha=0)
    lasso_reg_3 = Lasso(alpha=10)
    lasso_reg.fit(X,y)
    lasso_reg_2.fit(X,y)
    lasso_reg_3.fit(X,y)
    y_pred_lin = lin_reg.predict(X)
    y_pred_lasso = lasso_reg.predict(X)
    y_pred_lasso_2 = lasso_reg_2.predict(X)
    y_pred_lasso_3 = lasso_reg_3.predict(X)
    visualize(X[:,0],y_pred_lin,y_baseline1=y,y_baseline2=y_pred_lasso)
    visualize(X[:,0],y_pred_lasso,y_baseline1=y_pred_lasso_2,y_baseline2=y_pred_lasso_3)