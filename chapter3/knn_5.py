from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
def knn_classify(X,y):
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X,y)
    return knn_clf

if __name__=='__main__':
    train = pd.read_csv("datasets\\mnist_train.csv").drop(labels=["Unnamed: 0"],axis=1)
    test = pd.read_csv("datasets\\mnist_test.csv").drop(labels=["Unnamed: 0"],axis=1)
    train_X, train_y = np.asarray(train.drop(labels=["label"],axis=1)), np.asarray(train['label'])
    test_X, test_y = np.asarray(test.drop(labels=["label"],axis=1)), np.asarray(test['label'])
    train_y_5 = (train_y == 5)
    print(train_X.shape)
    knn_clf = knn_classify(train_X, train_y)
    print(knn_clf.predict([train_X[23894]]))
    
    # Image check
    import matplotlib
    import matplotlib.pyplot as plt
    image_45894 = train_X[23894].reshape(28,28)
    plt.imshow(image_45894, cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")
    plt.show()