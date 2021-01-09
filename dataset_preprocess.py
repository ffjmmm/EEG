import pandas as pd
import numpy as np

dataframe = pd.read_csv('./data_80.csv')
print(dataframe.shape)
# dataframe = dataframe.sample(frac=1)

train_data = dataframe[:300000]