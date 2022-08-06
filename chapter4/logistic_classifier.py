from numpy.lib.scimath import log
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_logit_proba(logit_model,X,y):
    X_new = np.linspace(0,3,1000).reshape(-1,1)
    y_proba = logit_model.predict_proba(X_new)
    plt.plot(X_new,y_proba[:,1],"g-",label="Iris virginica")
    plt.plot(X_new,y_proba[:,0],"b--",label="Not iris virginica")
    plt.legend()
    plt.show()

if __name__=='__main__':
    iris_dataset = load_iris()
    X = iris_dataset["data"][:,3:]
    y = (iris_dataset["target"]==2).astype(np.int)
    log_reg = LogisticRegression()
    log_reg.fit(X,y)
    plot_logit_proba(log_reg,X,y)
