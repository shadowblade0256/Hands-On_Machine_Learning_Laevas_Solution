# Need SSL Unverified Context support
from scipy.io import loadmat
import pandas as pd

def get_mnist(output=False):
    mnist = loadmat("datasets/mnist-original.mat")
    print(mnist['data'].shape)
    print(mnist['label'].shape)
    if output:
        pd.DataFrame(mnist["data"].T,
            columns=["features_"+str(i) for i in range(mnist['data'].shape[0])],).to_csv("datasets/mnist_data.csv")
        pd.DataFrame(mnist["label"].T,
            columns=["label"],).to_csv("datasets/mnist_label.csv")
    return mnist['data'].T,mnist['label'].T

if __name__=='__main__':
    X,y = get_mnist(output=True)