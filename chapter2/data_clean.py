import pandas as pd
from create_train_test import get_train_test
from data_init import load_housing_data

def fill_null(dirty_data):
    median = dirty_data["total_bedrooms"].median()
    dirty_data = dirty_data["total_bedrooms"].fillna(median)
    return dirty_data