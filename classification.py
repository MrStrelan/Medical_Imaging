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
import seaborn as sns

# Define a function to extract elements from lists
def extract_elements_from_list(string):
    # Convert the string to a list using ast.literal_eval
    lst = ast.literal_eval(string)
    # Return the extracted elements
    return lst[0], lst[1], lst[2]

def test_melanomas(data, trained=False):
    # Load the dataset from the CSV file
    dataset = pd.read_csv(data)
    
    # Apply the function to the column and add the extracted elements as new columns
    dataset[['color_h', 'color_s', 'color_v']] = dataset['color'].apply(extract_elements_from_list).apply(pd.Series)
    dataset[['border_h', 'border_s', 'border_v']] = dataset['border'].apply(extract_elements_from_list).apply(pd.Series)
    #print(dataset)
    
    # replace "." to missing value
    dataset=dataset.replace(".", np.nan)
    # drop all rows containing missing value
    dataset=dataset.dropna()

    # Create a new 'Type' column based on the diagnosis
    dataset['Type'] = dataset['diagnosis'].apply(lambda x: True if x in ['ACK', 'SCC', 'MEL'] else False)

    # Drop unnecessary columns
    columns_to_drop = ['id', 'diagnosis', 'smoker', 'color', 'border']
    dataset.drop(columns=columns_to_drop, inplace=True)

    # Separate features (X) and target variable (y) from the datasets
   
    X = dataset.drop(columns='Type')
    y = dataset['Type']
    
    X = f.normalizeFeatures(X)

    if trained == False:
        
        CLFS = {
            "linear_svc": svm.LinearSVC(max_iter=5000),
            "knn5": KNeighborsClassifier(n_neighbors=5),
            "DTC": DecisionTreeClassifier(random_state=0, max_depth=5)
        }

        results = {}
        fitted_models = {}
        
        
        X_train, X_test, y_train, y_test = f.splitDataIntoTrainTest(X, y, trained)
        
        for model_name, model in CLFS.items():
            model.fit(X_train, y_train)
            fitted_models[model_name] = model  # Store the fitted model
        
        #UNCOMENT TO SEE PAIRPLOT
        sns.pairplot(dataset, hue='Type', diag_kind='hist')
        plt.show()

        dumpFolder = ".\\testData\\"

        with open(dumpFolder + 'X_test.pkl', 'wb') as file1:
            pickle.dump(X_test, file1)

        with open(dumpFolder + 'y_test.pkl', 'wb') as file2:
            pickle.dump(y_test, file2)


        for modelName, model in fitted_models.items():
            
            with open(".\\models\\" + modelName + ".pkl", "wb") as file:
                pickle.dump(model, file)
  

        # Get feature scores for melanoma data set

        #feature_scores, selector = f.featureScores(X_train, y_train, k=2)
           
        #Uncoment to see features differences
        """
        plt.bar(np.arange(0,features), feature_scores, width=.2)
        plt.xticks(np.arange(0,features), list(X_train.columns), rotation='vertical')
        plt.show()
        """

        results[modelName] = f.evaluateTestData(X_test, y_test, fitted_models)  
        return results

    
    
    if trained == True:
        loaded_models = {}
        dumpFolder = ".\\testData"

        for model_name in os.listdir(".\\models"):
            filename = f"{model_name}"  # Specify the filename for each model

            with open(".\\models\\" + filename, "rb") as file:
                loaded_models[model_name] = pickle.load(file)
            
            print(model_name, "loaded")

        print("data prepared")
        results = {}
        results[model_name] = f.evaluateTestData(X, y, loaded_models)

        evaluation = f.predictData(X, loaded_models)

        return results, evaluation
