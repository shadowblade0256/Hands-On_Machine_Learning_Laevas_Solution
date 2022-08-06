# Load data from CSV file
import os
import pandas as pd
from fetch_data import HOUSING_PATH
from sklearn.model_selection import train_test_split

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,'housing.csv')
    return pd.read_csv(csv_path)

# Load housing_data from file
if __name__=='__main__':
    housing_data = load_housing_data()

