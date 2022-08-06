import pandas as pd
from sklearn.impute import SimpleImputer
from data_init import load_housing_data
from create_train_test import get_train_test
from pipeline_builder import generate_pipeline

# Create imputer
imputer = SimpleImputer(strategy="median")

# Apply the clean data to original one
def clean_data(raw_data_num):
    x = imputer.transform(raw_data_num)
    cleaned_data = pd.DataFrame(x,columns=raw_data_num.columns)
    return cleaned_data

if __name__=='__main__':
    # Load data
    raw_housing_data = load_housing_data()
    train_set, test_set = get_train_test(raw_housing_data)
    raw_housing_data_num = raw_housing_data.drop("ocean_proximity",axis=1)
    imputer.fit_transform(raw_housing_data_num)

def data_digitalizer(housing_data):
    pipeline = generate_pipeline(housing_data)
    housing_data_cleaned = pipeline.fit_transform(housing_data)
    return housing_data_cleaned