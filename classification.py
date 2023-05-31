import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

p2data = "dataExtracted.csv"

moles_raw = pd.read_csv(p2data)
#print(moles_raw.head())

columns_of_interest = ['diagnosis', "color", "symmetry", "compactness", "border", "smoker", "inheritance"]
moles_df = moles_raw.loc[:,columns_of_interest]

moles_data=moles_df.select_dtypes(np.number)
print(moles_data.head())