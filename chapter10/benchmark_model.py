import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
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
    model = LinearRegression()
    return model

if __name__=='__main__':
    X_train, y_train, X_test, y_test, X_val, y_val = get_data()
    model = get_model(X_train)
    model.fit(
        X=X_train,
        y=y_train,
    )
    print(mean_squared_error(y_val,model.predict(X_val)))
    pass