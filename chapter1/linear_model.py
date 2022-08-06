from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model as linear_model
from sklearn import neighbors as neighbors
from fetch_data import get_oecd
from fetch_data import get_gdp
from make_dataframe import prepare_country_stats

# Load the data
#get_gdp()
#get_oecd()
oecd_bli = pd.read_csv("datasets/lifesat/oecd_bli_2015.csv",thousands=',')
gdp_per_capita = pd.read_csv("datasets/lifesat/gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1',na_values='n/a')

# Prepare the data
country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)
x = np.c_[country_stats["2015"]]
y = np.c_[country_stats["Value"]]

# Data visualization
print(country_stats[:30])
plt.plot(country_stats[:30]["2015"])
plt.plot(country_stats[:30]["Value"])
plt.show()

# Select a linear model
lin_reg_model = neighbors.KNeighborsRegressor(n_neighbors=3)

# Train the model
lin_reg_model.fit(x,y)

# Make a prediction
print(lin_reg_model.get_params())
x_new = [[np.random.randint(30000)] for _ in range(10)]
print(lin_reg_model.predict(x_new))

plt.plot([x[0] for x in lin_reg_model.predict(x_new)])
plt.show()
pass