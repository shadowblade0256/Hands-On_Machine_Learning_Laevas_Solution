from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from pipeline_builder import generate_pipeline
import pandas as pd
import numpy as np

def dt_regression(housing_data,housing_prices):
    dt = DecisionTreeRegressor()
    dt.fit(housing_data,housing_prices)
    predicted_values = dt.predict(housing_data)
    return predicted_values,dt

if __name__=='__main__':
    housing_data = pd.read_csv("./datasets/trainset.csv")
    housing_prices = housing_data['median_house_value']
    housing_data.drop("median_house_value",axis=1,inplace=True)
    pipeline = generate_pipeline(housing_data)
    housing_data_cleaned = pipeline.fit_transform(housing_data)
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(housing_data_cleaned,housing_prices)

    # Test
    some_data = housing_data
    some_labels = housing_prices
    another_pipeline = generate_pipeline(some_data)
    some_data_prepared = another_pipeline.fit_transform(some_data)
    predicted_values = decision_tree.predict(some_data_prepared)
    dt_mse = mean_squared_error(housing_prices,predicted_values)
    dt_rmse = np.sqrt(dt_mse)
    print(dt_rmse)

    result = pd.DataFrame(data={'original':housing_prices,'predict':predicted_values})
    result.to_csv("result2.csv")
