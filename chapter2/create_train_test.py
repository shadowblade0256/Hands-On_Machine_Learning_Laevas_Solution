# Create train set & test set
import pandas as pd
from data_init import load_housing_data
from sklearn.model_selection import train_test_split

def get_train_test(raw_data,tsize=0.2):
    raw_data_with_id = raw_data.reset_index()
    return train_test_split(raw_data,test_size=tsize)

if __name__=='__main__':
    housing_data = load_housing_data()
    train_set, test_set = get_train_test(housing_data)
    #print(train_set[:10])
    pd.to_pickle(train_set,"datasets/trainset.pkl")
    pd.DataFrame.to_csv(train_set,"datasets/trainset.csv",index=False)
    pd.to_pickle(train_set,"datasets/testset.pkl")
    pd.DataFrame.to_csv(train_set,"datasets/testset.csv",index=False)