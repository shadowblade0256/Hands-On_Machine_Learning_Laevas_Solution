from sklearn.svm import SVC
import pandas as pd
import numpy as np

def linear_svm(housing_data,housing_prices):
    svm_reg = SVC(kernel="rbf",max_iter=100)
    svm_reg.fit(housing_data,housing_prices)
    predicted_values = svm_reg.predict(housing_data)
    return predicted_values, svm_reg