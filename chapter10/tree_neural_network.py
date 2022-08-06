import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
import matplotlib.pyplot as plt
from transformers import XLM_PRETRAINED_MODEL_ARCHIVE_LIST

class CombinedAttribAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, 3] / X[:, 6]
        population_per_household = X[:, 5] / X[:, 6]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 4] / X[:, 3]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def get_data(path="housing/housing.csv"):
    orig_data = pd.read_csv(path,na_values="")
    orig_data = orig_data.dropna(axis=0)
    orig_data = orig_data.drop(orig_data[orig_data["median_house_value"] > 500000].index)
    orig_data = orig_data.drop(orig_data[orig_data["median_income"] > 50].index)
    raw_data = orig_data.drop("median_house_value",axis=1)
    raw_labels = orig_data["median_house_value"]
    raw_data_num = raw_data.drop("ocean_proximity",axis=1)
    raw_data_text = raw_data["ocean_proximity"]
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attrib_adder", CombinedAttribAdder()),
        ("std_scaler",StandardScaler())
    ])
    text_pipeline = Pipeline([
        ("1hot_encoder",OneHotEncoder())
    ])
    data_pipeline = ColumnTransformer([
        ("num_attrib_pipeline",num_pipeline,list(raw_data_num)),
        ("txt_attrib_pipeline",text_pipeline,["ocean_proximity"])
    ])
    data = data_pipeline.fit_transform(raw_data)
    X_tv, X_test = data[5000:], data[:5000]
    X_train, X_val = X_tv[2500:], X_tv[:2500]
    y_tv, y_test = raw_labels[5000:], raw_labels[:5000]
    y_train, y_val = y_tv[2500:], y_tv[:2500]
    return X_train, y_train.values, X_val, y_val.values, X_test, y_test.values

def get_model(X):
    input_0 = keras.layers.Input(shape=X.shape[1:])
    hidden_0 = keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal")(input_0)
    hidden_1 = keras.layers.Dense(200, activation="relu", kernel_initializer="he_normal")(hidden_0)
    hidden_2 = keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal")(hidden_1)
    hidden_3 = keras.layers.Dense(50, activation="relu", kernel_initializer="he_normal")(hidden_2)
    concat = keras.layers.Concatenate()([input_0,hidden_3])
    output_0 = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_0],outputs=[output_0])
    print(input_0.shape)
    return model

if __name__=='__main__':
    keras.backend.set_floatx("float64")

    X_train, y_train, X_test, y_test, X_val, y_val = get_data()

    model = get_model(X_train)
    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(learning_rate=5e-3)
    )
    checkpoint_cb = keras.callbacks.ModelCheckpoint("tree_housing_nn.h5",save_best_only=True)
    early_stopping_es = keras.callbacks.EarlyStopping(monitor="val_loss",patience=3,restore_best_weights=True)
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb]
    )

    model = keras.models.load_model("tree_housing_nn.h5")
    model.evaluate(
        x=X_test,
        y=y_test,
        return_dict=True
    )
    
    print(model.predict(X_train[:3]))
    print(history.params)

    plt.plot([(t+1) for t in history.epoch],history.history["loss"],label="Train Loss")
    plt.plot([(t+1) for t in history.epoch],history.history["val_loss"],label="Validation Loss")
    plt.title("Train & Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.xticks(np.arange(0,51,step=2))
    plt.legend()
    plt.show()

    pass