from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

def rf_regression(housing_data,housing_prices):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(housing_data,housing_prices)
    predicted_values = rf_reg.predict(housing_data)
    return predicted_values, rf_reg