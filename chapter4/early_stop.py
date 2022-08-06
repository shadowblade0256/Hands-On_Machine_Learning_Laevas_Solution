from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def early_stop_sgd(X,y,max_epoch=300):
    # Data preparation
    poly_and_scaler = Pipeline([
        ("poly_adder",PolynomialFeatures(degree=4)),
        ("std_scaler",StandardScaler())
    ])
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2)
    X_train_scaled = poly_and_scaler.fit_transform(X_train)
    X_val_scaled = poly_and_scaler.fit_transform(X_val)

    sgd_reg = SGDRegressor(max_iter=10,tol=np.infty,warm_start=True,
        penalty=None,learning_rate="constant",eta0=0.0005)
    
    minimum_val_error = float("inf")
    best_epoch = None
    best_model = None
    for epoch in range(max_epoch):
        sgd_reg.fit(X_train,y_train)    # Continues where it left off
        y_val_pred = sgd_reg.predict(X_val)
        val_error = mean_squared_error(y_val,y_val_pred)
        if minimum_val_error > val_error:
            minimum_val_error = val_error
            best_epoch = epoch
            best_model = clone(sgd_reg)
    print(best_model.fit_intercept)
    return best_model, best_epoch

def generate_data(m=300):
    X = np.random.uniform(low=-30,high=30,size=(m,1))
    y = 40*(X**4)-3.5*(X**4)+4*X
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
    model, _ = early_stop_sgd(X,y)
    plot_learning_curves(model,X,y)