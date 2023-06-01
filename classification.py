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

<<<<<<< HEAD
<<<<<<< HEAD
moles_df = pd.read_csv(p2data)
#print(moles_df.head())
#print(type(moles_df))

# replace "." to missing value
moles_df=moles_df.replace(".", np.nan)
# drop all rows containing missing value
moles_df=moles_df.dropna()
# Drop Image Id
moles_df = moles_df.drop(['id'],axis=1)
# Drop Smoking because we don't have enough data on smoking
moles_df = moles_df.drop(['smoker'],axis=1)
=======
# Read the data from the CSV file into a pandas DataFrame
moles_raw = pd.read_csv(p2data)
=======
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
>>>>>>> aaf081375771e11f5417844bc73ce7e02cfad36f

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

>>>>>>> aaf081375771e11f5417844bc73ce7e02cfad36f
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

<<<<<<< HEAD
<<<<<<< HEAD
diagnosis_groups = moles_df.groupby("diagnosis")

datasets = {}

dropped_column = moles_df["diagnosis"]

# Iterate over each group and create a new DataFrame without the "diagnosis" column for normalisation
for diagnosis, group_df in diagnosis_groups:
    dropped_column = moles_df["diagnosis"]
    datasets[diagnosis] = moles_df.drop("diagnosis", axis=1)
    

# Access the individual datasets using the diagnosis as the key
for diagnosis, dataset in datasets.items():

    print(f"Dataset for diagnosis '{diagnosis}':")
    #Normalising our data
    moles_data = f.normalizeFeatures(dataset)
    
    #result_df = pd.DataFrame(False, index=dropped_column.index, columns=dropped_column.columns)
    # Set values to True where "diagnosis" matches the given value
    #result_df["result"] = diagnosis["diagnosis"] == given_value

    # Print the resulting DataFrame
    #print("Here",result_df)
    columns_with_features = list(moles_data.columns[:])
    moles_data['Type'] = diagnosis
    if diagnosis != "MEL":
        X_train, X_test, y_train, y_test = f.splitDataIntoTrainTest(moles_data[columns_with_features], moles_data.iloc[:, 9])
    print(type(X_train))
    print(X_train)
    X_train_with_Y = X_train.copy()
    X_train_with_Y[diagnosis] = y_train
    sns.pairplot(X_train_with_Y, hue=diagnosis, height=3, diag_kind="hist");
    plt.show()
    """
    # Get feature scores for melanoma data set
    feature_scores_ker, selector_ker = featureScores(X_train_ker, y_train_ker, k=2)

    # Get no. of features
    features_ker = len(feature_scores_ker)

    # Visualize feature scores
    plt.bar(np.arange(0,features_ker), feature_scores_ker, width=.2)
    plt.xticks(np.arange(0,features_ker), list(X_train_ker.columns), rotation='vertical')
    plt.show()

    # Select the two best features based on the selector
    X_train_ker_adj = selector_ker.transform(X_train_ker)
    X_test_ker_adj = selector_ker.transform(X_test_ker)
    print("\n")
    """




=======
# Select columns with numerical values from the DataFrame
moles_data = moles_df.select_dtypes(np.number)

=======
# Select columns with numerical values from the DataFrame
moles_data = moles_df.select_dtypes(np.number)

>>>>>>> aaf081375771e11f5417844bc73ce7e02cfad36f
# Normalize the features in the DataFrame using a custom function normalizeFeatures
moles_data = f.normalizeFeatures(moles_data)

# Get the column names of the features
columns_with_features = list(moles_data.columns[:])

# Split the data into training and testing sets using a custom function splitDataIntoTrainTest
X_train_mel, X_test_mel, y_train_mel, y_test_mel = f.splitDataIntoTrainTest(moles_data[columns_with_features], moles_data.iloc[:, 0])
<<<<<<< HEAD
>>>>>>> aaf081375771e11f5417844bc73ce7e02cfad36f
=======
>>>>>>> aaf081375771e11f5417844bc73ce7e02cfad36f
