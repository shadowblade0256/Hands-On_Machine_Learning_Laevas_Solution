import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
import matplotlib.pyplot as plt
import gzip

def parse_gzip_file(image_file_name: str, label_file_name: str):
    keras.datasets.fashion_mnist.load_data()
    with gzip.open(label_file_name, "rb") as lbfile:
        labels = np.frombuffer(lbfile.read(), dtype=np.uint8, offset=8)
    with gzip.open(image_file_name, "rb") as imgfile:
        images = np.frombuffer(imgfile.read(), dtype=np.uint8,offset=16).reshape((len(labels),28,28))
    labels = keras.utils.to_categorical(labels)
    return images, labels

def get_data():
    X_train_full, y_train_full = parse_gzip_file(
        "fashion/train-images-idx3-ubyte.gz",
        "fashion/train-labels-idx1-ubyte.gz"
    )
    X_test, y_test = parse_gzip_file(
        "fashion/t10k-images-idx3-ubyte.gz",
        "fashion/t10k-labels-idx1-ubyte.gz"
    )
    X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    class_names = ["T-shirts/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test), class_names

def get_model():
    input_0 = keras.layers.Input(shape=[28,28])
    flatten_0 = keras.layers.Flatten(input_shape=[28,28])(input_0)
    dense_0 = keras.layers.Dense(300,activation="relu")(flatten_0)
    dense_1 = keras.layers.Dense(100,activation="relu")(dense_0)
    dense_2 = keras.layers.Dense(10,activation="softmax")(dense_1)
    model = keras.Model(inputs=[input_0],outputs=[dense_2])
    return model

if __name__=='__main__':
    train, valid, test, class_names = get_data()
    model = get_model()
    model.compile(
        loss = keras.losses.CategoricalCrossentropy(),
        optimizer = keras.optimizers.SGD(),
        metrics = [keras.metrics.CategoricalAccuracy()]
    )

    history = model.fit(
        x = train[0],
        y = train[1],
        epochs = 20,
        validation_data = (valid[0], valid[1]),
    )

    print(history.epoch)
    print(history.history)
    
    plt.plot([(t+1) for t in history.epoch],history.history["loss"],label="Train Loss")
    plt.plot([(t+1) for t in history.epoch],history.history["categorical_accuracy"],label="Train ACC")
    plt.plot([(t+1) for t in history.epoch],history.history["val_loss"],label="Validation Loss")
    plt.plot([(t+1) for t in history.epoch],history.history["val_categorical_accuracy"],label="Validation ACC")
    plt.title("Train & Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Categorical Accuracy")
    plt.xticks(np.arange(0,21,step=1))
    plt.legend()
    plt.show()

    model.evaluate(
        x = test[0],
        y = test[1],
        return_dict = True,
    )