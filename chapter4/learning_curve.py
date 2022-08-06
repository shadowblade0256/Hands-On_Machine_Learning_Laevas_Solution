from os import X_OK
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def generate_data(m=300):
    X = np.random.normal(loc=0,scale=1,size=(m,1))
    y = 4.5*(X**2)+2.5*X+3+np.random.normal(loc=0,scale=1,size=(m,1))
    return X, y

# Add polynomial terms as new features
# e.g.: X(a,b) -> X(a,b,a^2,ab,b^2)
def add_poly_features(X):
    poly_transformer = PolynomialFeatures(degree=2,include_bias=False)
    X_new = poly_transformer.fit_transform(X)
    return X_new

def plot_learning_curves(model,X,y):
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    train_mse, val_mse = [],[]
    # Read each data entry xi, fit (x[0:i],y[0:i]), calculate their MSE to original data
    # Then plot the relations between data size and MSE
    for i in range(1,len(X_train)):
        model.fit(X_train[:i],y_train[:i])
        y_train_predict = model.predict(X_train[:i])
        y_val_predict = model.predict(X_val[:i])
        train_mse.append(mean_squared_error(y_train[:i],y_train_predict))
        val_mse.append(mean_squared_error(y_val[:i],y_val_predict))
        print(model.intercept_,model.coef_)
    plt.plot(np.sqrt(train_mse),"r-+",linewidth=2,label="Train")
    plt.plot(np.sqrt(val_mse),"b-",linewidth=2,label="Validation")
    plt.legend()
    plt.show()

if __name__=="__main__":
    X, y = generate_data()
    X_bi = add_poly_features(X)
    bi_reg = LinearRegression()
    plot_learning_curves(bi_reg,X_bi,y)
