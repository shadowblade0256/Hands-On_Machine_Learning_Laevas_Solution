from decision_tree import dt_regression
from linear_regression import linear_regression
from random_forest import rf_regression
from svm import linear_svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from data_clean_sklearn import data_digitalizer
from pprint import pprint
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np

def rf_grid_search(model,inp,label):
    param_grid = [
        {'n_estimators':[3,10,30],'max_features':[2,4,6,8]},
        {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}
    ]
    grid_searched = GridSearchCV(model,param_grid,cv=5,scoring='neg_mean_squared_error')
    grid_searched.fit(inp,label)
    return grid_searched.best_estimator_,grid_searched

def features_analyze(model,inp):
    features_importances = model.feature_importances_
    extra_attribs = ["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
    label_encoder = LabelBinarizer()
    label_encoder.fit(inp["ocean_proximity"])
    cat_one_hot_attribs = list(label_encoder.classes_)
    attributes = list(housing_data)+extra_attribs+cat_one_hot_attribs
    print(sorted(zip(features_importances,attributes),reverse=True))

if __name__=='__main__':
    housing_data = pd.read_csv("./datasets/trainset.csv")
    housing_prices = housing_data['median_house_value']
    housing_data.drop(["median_house_value"],axis=1,inplace=True)
    housing_data_c = data_digitalizer(housing_data)
    rf_result,rf = rf_regression(housing_data_c, housing_prices)
    optimized_rf, opt_info = rf_grid_search(rf, housing_data_c, housing_prices)
    features_analyze(optimized_rf, housing_data)
    