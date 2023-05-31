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

# Read the data from the CSV file into a pandas DataFrame
moles_raw = pd.read_csv(p2data)

# Print the first few rows of the DataFrame and its type
print(moles_raw.head())
print(type(moles_raw))

# Select columns of interest from the DataFrame
columns_of_interest = ["color", "symmetry", "compactness", "border", "diagnosis", "smoker", "inheritance"]
moles_df = moles_raw.loc[:, columns_of_interest]

# Replace "." with missing values (NaN) in the DataFrame
moles_df = moles_df.replace(".", np.nan)

# Drop rows containing missing values
moles_df = moles_df.dropna()

# Convert string columns to lists of floats

# Convert 'border' column to lists of floats using ast.literal_eval
moles_df['border'] = moles_df['border'].apply(lambda x: ast.literal_eval(x))

# Convert 'color' column to lists of floats using ast.literal_eval
moles_df['color'] = moles_df['color'].apply(lambda x: ast.literal_eval(x))

# Split the 'border' column into separate columns ('border_h', 'border_s', 'border_v')
border_split = moles_df['border'].apply(pd.Series)
border_split.columns = ['border_h', 'border_s', 'border_v']

# Split the 'color' column into separate columns ('color_h', 'color_s', 'color_v')
color_split = moles_df['color'].apply(pd.Series)
color_split.columns = ['color_h', 'color_s', 'color_v']

# Merge the split columns ('border' and 'color') with the original DataFrame
moles_df = pd.concat([moles_df.drop(['border', 'color'], axis=1), border_split, color_split], axis=1)

# Select columns with numerical values from the DataFrame
moles_data = moles_df.select_dtypes(np.number)

# Normalize the features in the DataFrame using a custom function normalizeFeatures
moles_data = f.normalizeFeatures(moles_data)

# Get the column names of the features
columns_with_features = list(moles_data.columns[:])

# Split the data into training and testing sets using a custom function splitDataIntoTrainTest
X_train_mel, X_test_mel, y_train_mel, y_test_mel = f.splitDataIntoTrainTest(moles_data[columns_with_features], moles_data.iloc[:, 0])
