import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
import functions as f
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
import pickle
import json

def test_melonomas(p2data, trained = False):

    templist = []
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

    diagnosis_groups = moles_df.groupby("diagnosis")

    datasets = {}

    dropped_column = moles_df["diagnosis"]

    # Iterate over each group and create a new DataFrame without the "diagnosis" column for normalisation
    for diagnosis, group_df in diagnosis_groups:
        dropped_column = moles_df["diagnosis"]
        datasets[diagnosis] = moles_df.drop("diagnosis", axis=1)
        
        
        #Select and build classifiers
        CLFS = {
            "linear_svc": svm.LinearSVC(max_iter = 5000),
            "knn5": KNeighborsClassifier(n_neighbors=5),
            "DTC": DecisionTreeClassifier(random_state=0, max_depth=5)
        }
        results = {}

        # Access the individual datasets using the diagnosis as the key
        for diagnosis, dataset in datasets.items():

            #print(f"Dataset for diagnosis '{diagnosis}':")
            #Normalising our data
            moles_data = f.normalizeFeatures(dataset)
            #Extracting values that we will build classifier on
            columns_with_features = list(moles_data.columns[:])
            #Adding back column with type deleted for normalisation
            moles_data['Type'] = dropped_column
            #Setting Type value to true if mole matches Mole we are building model now for
            moles_data['Type'] = moles_data['Type'].apply(lambda x: True if x == diagnosis else False)
            X_train, X_test, y_train, y_test = f.splitDataIntoTrainTest(moles_data[columns_with_features], moles_data.iloc[:, 9], trained = False)

            X_train_with_Y = X_train.copy()
            X_train_with_Y[diagnosis] = y_train

            """Uncomment to see allfeatures corelation
            #sns.pairplot(X_train_with_Y, hue=diagnosis, height=3, diag_kind="hist");
            #plt.show()
            """

            # Get feature scores for melanoma data set
            feature_scores, selector = f.featureScores(X_train, y_train, k=2)
            

            # Visualize feature scores
            #Uncoment to see features differences
            """
            plt.bar(np.arange(0,features), feature_scores, width=.2)
            plt.xticks(np.arange(0,features), list(X_train.columns), rotation='vertical')
            plt.show()
            """
            # Select the two best features based on the selector
            X_train_adj = selector.transform(X_train)
            X_test_adj = selector.transform(X_test)
            
            #Uncomment to see cross vadidation of classifires
            #f.crossValidate(X_train_adj, y_train, CLFS))
            if trained == False:
                #Evaluate the results - TEST DATA
                CLFS_trained = {
                    "linear_svc": svm.LinearSVC(max_iter = 5000).fit(X_train_adj, y_train),
                    "knn5": KNeighborsClassifier(n_neighbors=5).fit(X_train_adj, y_train),
                    "DTC": DecisionTreeClassifier(random_state=0, max_depth=5).fit(X_train_adj, y_train)
                }

              

                for model_name, model in CLFS_trained.items():
                    filename = f"{model_name}.pkl"  # Specify the filename for each model

                    with open(".\\models\\" + filename, "wb") as file:
                        pickle.dump(model, file)


                results[diagnosis] = f.evaluateTestData(X_test_adj, y_test, CLFS_trained)
            
            if trained == True:
                    
                loaded_models = {}
                results = {}

             

                for model_name in os.listdir(".\\models"):
                    filename = f"{model_name}"  # Specify the filename for each model

                    with open(".\\models\\" + filename, "rb") as file:
                        loaded_models[model_name] = pickle.load(file)

                for model_name, model in loaded_models.items():
                    model_name = model_name[:-4]
                    #print(model.predict(X_test_adj))
                    results[diagnosis] = f.evaluateTestData(X_test_adj, y_test, loaded_models)
                    templist.append(results)
                
                
        


    #print(miscellaniousDict)
    """Uncomment to see main parameters corelation
    plt.figure(figsize=(10, 5))
    if diagnosis == "NEV":
        sns.scatterplot(x="inheritance", y="color_h", data=X_train_with_Y, hue=diagnosis)
    elif diagnosis == "ACK":
        sns.scatterplot(x="inheritance", y="border_h", data=X_train_with_Y, hue=diagnosis)
    elif diagnosis == "SCC":
        sns.scatterplot(x="inheritance", y="border_h", data=X_train_with_Y, hue=diagnosis)
    elif diagnosis == "BCC":
        sns.scatterplot(x="border_v", y="color_v", data=X_train_with_Y, hue=diagnosis)
    elif diagnosis == "MEL":
        sns.scatterplot(x="symmetry", y="border_s", data=X_train_with_Y, hue=diagnosis)
    plt.show()
    """
    for el in templist:
        print(el)
    return results