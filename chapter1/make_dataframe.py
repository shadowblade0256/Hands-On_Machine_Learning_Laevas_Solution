import numpy as np
import pandas as pd

def prepare_country_stats(oecd,gdp_per_capita):
    gdp_data = pd.DataFrame(gdp_per_capita[['Country','2015']])
    oecd_data = pd.DataFrame(oecd[['Country','Value']])
    oecd_data = oecd_data.merge(gdp_data,how="left",left_on="Country",right_on="Country")
    oecd_data = oecd_data.dropna(axis=0)
    return oecd_data