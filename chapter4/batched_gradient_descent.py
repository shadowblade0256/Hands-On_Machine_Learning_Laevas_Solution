from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Custom version of Batched Gradient Descent (BGD)
class CustomBGD(BaseEstimator):
    def __init__(self,degree=1,learning_rate=0.02,n_iterations=1000):
        self.degree = degree    # Degree of polynomial model
        self.learning_rate = learning_rate  # Learning rate (eta), describing how fast the gradient can descend at most
        self.n_iterations = n_iterations    # Number of iterations
        self.theta = None                   # Factors of polynomial model. (theta0,theta1,...,theta_degree)
    def fit(self,X,y):
        # Add polynomial features first if not linear
        if self.degree>1:
            poly_add = PolynomialFeatures(degree=self.degree)
            X = poly_add.fit_transform(X)
        m = X.shape[0]              # number of samples
        n_features = self.degree+1  # number of features
        self.theta = np.random.randn(n_features,1)  # Initially, thetas are randomly set (following normal(0,1))
        # Find gradients for all directions, update thetas
        for i in range(self.n_iterations):
            gradients = (2/m) * X.T.dot(X.dot(self.theta)-y)
            self.theta = self.theta - self.learning_rate * gradients
    def predict(self,X):
        if self.degree>1:
            poly_add = PolynomialFeatures(degree=self.degree)
            X = poly_add.fit_transform(X)
        y = np.matmul(X,self.theta)
        return y

def generate_data(m=500):
    X = np.random.normal(loc=0,scale=1,size=(m,1))
    y = 4.5*(X**2)+2.5*X+3+np.random.normal(loc=0,scale=1,size=(m,1))
    return X, y

def plot_learning_curves(model,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    train_mse, val_mse = [],[]
    for i in range(1,len(X_train)):
        model.fit(X_train[:i],y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_val_predict = model.predict(X_val[:i])
        train_mse.append(mean_squared_error(y_train[:i],y_train_predict))
        val_mse.append(mean_squared_error(y_val[:i],y_val_predict))
        print(model.theta)
    plt.plot(np.sqrt(train_mse),"r-+",linewidth=2,label="Train")
    plt.plot(np.sqrt(val_mse),"b-",linewidth=2,label="Validation")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X, y = generate_data()
    bgd_reg = CustomBGD(degree=2)
    plot_learning_curves(bgd_reg,X,y)