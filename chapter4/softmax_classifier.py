from numpy.lib.scimath import log
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_logit_proba(logit_model,X,y):
    y_pred = logit_model.predict(X)
    X_true = X[y_pred==1]
    X_false = X[y_pred==0]
    plt.scatter(X_true[:,0],X_true[:,1],label="Iris virginica")
    plt.scatter(X_false[:,0],X_false[:,1],label="Not iris virginica")
    plt.legend()
    plt.show()

if __name__=='__main__':
    iris_dataset = load_iris()
    X = iris_dataset["data"][:,2:]
    y = (iris_dataset["target"]==2).astype(np.int)
    log_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=1)
    log_reg.fit(X,y)
    print("Intercept:",log_reg.intercept_)
    print("Coefficient:",log_reg.coef_)
    plot_logit_proba(log_reg,X,y)
