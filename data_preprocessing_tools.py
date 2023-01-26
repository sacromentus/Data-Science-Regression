# Gerald Sufleta

# Data Preprocessing Tools

## Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

"""-in Python we read row,column in th e brackets so it is [row,column]

-We use a ':' to draw everything from the range.


"""

print(X)

print(y)

"""## Taking care of missing data"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

"""-Use the average of the other variables to fill in the missing data

-Call SimpleImputer class first argument is which values to replace, we use NaN, second value is what to replace those values with, we use mean/average value

-fit will calculate the average values

-imputer returns a filled in version with what fit calculated
"""

print(X)

"""### Encoding the Independent Variable

-We use "One Hot" encoding to give each country a vector so no order is implied by the categorical data
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

"""-transformers arguments: 'encoder' means you want to encode, class name to encode, and column you want to encode. ColumnTransforme
r(encoder, class, column indeX)

-passthrough means keep columns that are not encoded, here Age & Salary
"""

print(X)

"""### Encoding the Dependent Variable

Transform the dependent variable for Purchased to a binary No = 0 Yes = 1
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

"""## Splitting the dataset into the Training set and Test set

-We split the dataset into a training set and test set with train_test_split

-train_test_split returns four variables: X and y training set + X and y test set
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

"""## Feature Scaling

We do feature scaling always after we split the dataset

-Standardization for feature scaling always works b/c it creates a std distribution whereas normalization needs to be a normal distribution already

-we first use fit_transform to calculate the distribution used for standardization then transform the set with the values calculated
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)

print(X_test)
