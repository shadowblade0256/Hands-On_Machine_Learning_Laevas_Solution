from decision_tree import dt_regression
from linear_regression import linear_regression
from random_forest import rf_regression
from svm import linear_svm
from sklearn.model_selection import cross_val_score
from data_clean_sklearn import data_digitalizer
import pandas as pd
import numpy as np

def model_cross_validation(model,inp,label):
    scores = cross_val_score(model,inp,label,scoring="neg_mean_squared_error",cv=10)
    rmse_scores = np.sqrt(-scores)
    print("Scores: ",rmse_scores)
    print("Mean: ",rmse_scores.mean())
    print("Standard deviation: ",rmse_scores.std())

if __name__=='__main__':
    housing_data = pd.read_csv("./datasets/trainset.csv")
    housing_data = housing_data
    housing_prices = housing_data['median_house_value']
    housing_data.drop("median_house_value",axis=1,inplace=True)
    housing_data_c = data_digitalizer(housing_data)
    lin_result,lin_reg = linear_regression(housing_data_c,housing_prices)
    # dt_result,dt = dt_regression(housing_data_c,housing_prices)
    # rf_result,rf = rf_regression(housing_data_c, housing_prices)
    svm_result,svm = linear_svm(housing_data_c, housing_prices)

    model_cross_validation(lin_reg,housing_data_c,housing_prices)
    # model_cross_validation(dt,housing_data_c,housing_prices)
    # model_cross_validation(rf,housing_data_c,housing_prices)
    model_cross_validation(svm,housing_data_c,housing_prices)