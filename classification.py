import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import functions as f

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p2data = "dataExtracted.csv"

moles_raw = pd.read_csv(p2data)
print(moles_raw.head())
print(type(moles_raw))

columns_of_interest = ["color", "symmetry", "compactness", "border", "diagnosis", "smoker", "inheritance"]
moles_df = moles_raw.loc[:,columns_of_interest]

# replace "." to missing value
moles_df=moles_df.replace(".", np.nan)
# drop all rows containing missing value
moles_df=moles_df.dropna()

# Convert string columns to lists of floats
moles_df['border'] = moles_df['border'].apply(lambda x: ast.literal_eval(x))
moles_df['color'] = moles_df['color'].apply(lambda x: ast.literal_eval(x))

# Splitting 'border' column
border_split = moles_df['border'].apply(pd.Series)
border_split.columns = ['border_h', 'border_s', 'border_v']

# Splitting 'color' column
color_split = moles_df['color'].apply(pd.Series)
color_split.columns = ['color_h', 'color_s', 'color_v']

# Merging the split columns border and column with the original DataFrame
moles_df = pd.concat([moles_df.drop(['border', 'color'], axis=1), border_split, color_split], axis=1)

# Makes an array just with columns that have numerical values
moles_data=moles_df.select_dtypes(np.number)

moles_data = f.normalizeFeatures(moles_data)

columns_with_features = list(moles_data.columns[:])

X_train_mel, X_test_mel, y_train_mel, y_test_mel = f.splitDataIntoTrainTest(moles_data[columns_with_features], moles_data.iloc[:, 0])



