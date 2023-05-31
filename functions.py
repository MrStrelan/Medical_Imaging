from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd

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

def splitDataIntoTrainTest(X, y):

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

    # Return the result
    return X_train, X_test, y_train, y_test