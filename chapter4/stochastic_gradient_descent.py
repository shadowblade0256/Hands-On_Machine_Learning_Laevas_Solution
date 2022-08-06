from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class CustomSGD(BaseEstimator):
    def __init__(self,degree=1,n_epochs=50,t0=5,t1=50,random_state=None):
        self.degree = degree            # Degree of polynomial model
        self.learning_rate = None       # Learning rate (eta), deprecated here
        self.n_epochs = n_epochs        # Number of epochs (iterations)
        self.theta = None               # Factors of polynomial
        self.t0 = t0                    # Hyperparameters for learning rate 
        self.t1 = t1
        self.random_state = random_state
    def fit(self,X,y):
        np.random.seed(self.random_state)
        if self.degree>1:
            poly_add = PolynomialFeatures(degree=self.degree)
            X = poly_add.fit_transform(X)
        def learning_schedule(t):
            return self.t0/(t+self.t1)
        std_scaler = StandardScaler()
        X = std_scaler.fit_transform(X)
        n_features = self.degree + 1
        m = X.shape[0]
        self.theta = np.random.randn(n_features,1)
        for epoch in range(self.n_epochs):
            for i in range(m):
                random_index = np.random.randint(m)
                xi = X[random_index:random_index+1]
                yi = y[random_index:random_index+1]
                gradients = 2 * xi.T.dot(xi.dot(self.theta)-yi)
                learning_rate = learning_schedule(epoch*m+i)
                self.theta = self.theta - learning_rate * gradients
    def predict(self,X):
        np.random.seed(self.random_state)
        std_scaler = StandardScaler()
        X = std_scaler.fit_transform(X)
        if self.degree>1:
            poly_add = PolynomialFeatures(degree=self.degree)
            X = poly_add.fit_transform(X)
        y = np.matmul(X,self.theta)
        return y

def generate_data(m=300):
    X = np.random.normal(loc=0,scale=1,size=(m,1))
    y = 4.5*(X**3)+2.5*X+3+np.random.normal(loc=0,scale=1,size=(m,1))
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
    plt.plot(np.sqrt(train_mse),"r-+",linewidth=2,label="Train")
    plt.plot(np.sqrt(val_mse),"b-",linewidth=2,label="Validation")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X, y = generate_data()
    bgd_reg = CustomSGD(degree=3)
    plot_learning_curves(bgd_reg,X,y)