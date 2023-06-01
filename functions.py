from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

def normalizeFeatures(features):

    """
    Normalizes features to a common scale using Sklearn preproccessing module.
    :features: pandas dataframe with selected features
    :return: dataframe with scaled features
    """

    # Fit scaler on the data
    scaler = preprocessing.StandardScaler().fit(features)

    # Apply the scaler to the data
    new_features = scaler.transform(features) # Returns 2D numpy array

    # Transform the numpy array back to DF

    new_features = pd.DataFrame(data = new_features, columns = list(features.columns))

    return new_features

def splitDataIntoTrainTest(X, y, trained):

    """
    Wrapper around the scikit function train_test_split.
    The goal of this function is to split properly a given dataset into training and test data.
    :X: DF containing only features
    :y: pd series containing binary values
    :return: dataframes and series splitted according to the given criteria.
    """

    # Split the given data according to given criteria
    # * random state --> for reproducibility
    # * stratify --> makes sure that the distribution of cancer is in each equal
    if trained == False:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=1, shuffle=True, stratify=y)

    # Return the result
    return X_train, X_test, y_train, y_test

def featureScores(X_train, y_train, k):
    """
    This fucntion returns the selector object and importance scores and for each feature 
    for a univariate, filter-based feature selection.
    :X_train: Set of input variables for testing
    :y_train: Set of output variables for testing
    :k: Number of features to be selected (integer)
    """
        
    selector = SelectKBest(mutual_info_classif, k=k) # Selecting top k features using mutual information for feature scoring
    selector.fit(X_train, y_train) # fit selector to data

    # Retrieve scores
    scores = selector.scores_

    return scores, selector

def crossValidate(X_train, y_train, clfs):

    # Prepare cross-validation
    # * Specify K - industry standard is 5 - 10
    K = 5
    cv = StratifiedShuffleSplit(n_splits=K, test_size=0.2, random_state=1)

    # Build a dataframe where you will save the results of cross-validation
    # * Define metrics that will be measured
    metrics = ["accuracy", "precision", "recall", "roc_auc"]
    header = ["classifier_name"] + metrics
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)

    # Compute the results
    for name, clf in clfs.items():

        # Get the results for each metric as dict
        result_dict = cross_validate(clf, X_train, y_train, cv=cv, scoring = metrics)

        # Condense the results using their mean and save it to the list
        result = []
        for metric_name in metrics:
            key =  f"test_{metric_name}"
            value = np.mean(result_dict[key])
            result.append(value)
        
        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)

    return results

def evaluateTestData(X_test, y_true, clfs):

    # Build a dataframe where you will save the results
    # * Define metrics that will be measured
    metrics = ["accuracy", "precision", "recall"]
    header = ["classifier_name"] + metrics
    # * Build the empty dataframe
    results = pd.DataFrame(columns = header)
    # Compute the results
    for name, clf in clfs.items():
        
        # Get the results
        y_pred = clf.predict(X_test)

        # Compute the metrics
        # * Accuracy
        acc = accuracy_score(y_true, y_pred)
        
        # * Precision
        prec = precision_score(y_true, y_pred)

        # * Recall
        rec = recall_score(y_true, y_pred)

        result = [acc, prec, rec]

        # Create a new record representing the given classifier and corresponding metrics
        new_record = pd.DataFrame([[name] + result], columns = header)
        results = results.append(new_record, ignore_index=True)
    
    return results