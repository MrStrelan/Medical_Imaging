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




