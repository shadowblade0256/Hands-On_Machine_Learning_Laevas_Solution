from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataframe_select import DataFrameSelector
from compound_attrib_adder import CombinedAttributeAdder

# 打个小补丁
LabelBinarizer.__fit_transform = LabelBinarizer.fit_transform
LabelBinarizer.fit_transform = lambda self, X, y=None, **fit_params: LabelBinarizer.__fit_transform(self, X)

def generate_pipeline(housing_data):
    housing_num = housing_data.drop("ocean_proximity",axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    num_pipeline = Pipeline([
        ('selector',DataFrameSelector(num_attribs)),
        ('imputer',SimpleImputer(strategy="median")),
        ('attribs_adder',CombinedAttributeAdder()),
        ('std_scaler',StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('selector',DataFrameSelector(cat_attribs)),
        ('label_binarizer',LabelBinarizer()),
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline',num_pipeline),
        ('cat_pipeline',cat_pipeline)
    ])

    return full_pipeline

if __name__=='__main__':
    import pandas as pd
    housing_data = pd.read_csv("./datasets/trainset.csv")
    data_pipeline = generate_pipeline(housing_data)
    output = data_pipeline.fit_transform(housing_data)
    print(output)