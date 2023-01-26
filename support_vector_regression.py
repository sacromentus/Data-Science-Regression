# Gerald Sufleta

# Support Vector Regression (SVR)

## Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)

print(y)

y = y.reshape(len(y),1)

print(y)

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print(X)

print(y)

"""## Training the SVR model on the whole dataset

This code segment is importing the StandardScaler class from the sklearn.preprocessing module.

Then it creates two instances of the StandardScaler class, sc_X and sc_y, which will be used to standardize the independent variable X and the dependent variable y respectively.
The fit_transform method is then called on X and y, which standardizes the data by centering and scaling.
"""

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

"""## Predicting a new result

The method reshape(-1,1) is used to reshape the output into a 2-dimensional array with one column and as many rows as necessary to accommodate all of the elements. The -1 tells numpy to infer the number of rows based on the number of elements and the number of columns.
"""

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

"""## Visualising the SVR results

-This code segment is creating a scatter plot of the original, non-standardized data for the independent variable (X) and the dependent variable (y).

-It uses the inverse_transform method from the StandardScaler objects sc_X and sc_y to convert the data back to its original scale.

-Then it plots a line chart of the predicted values using the regressor.predict() method on the standardized X data.

-The inverse_transform method is also used on the predicted values to bring them back to the original scale.

-Finally, the plt.show() is used to display the plot.
"""

plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'indianred')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color = 'mediumblue')
plt.title('Position Salary Prediction')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Visualising the SVR results (for higher resolution and smoother curve)

-This code segment is creating an array of values for the independent variable "X_grid" using the arange function and the min and max values of the non-standardized X data.

-It uses the inverse_transform method from the StandardScaler object sc_X to convert the data back to its original scale.

-The X_grid array is reshaped to a 2-dimensional array with one column and as many rows as necessary to accommodate all of the elements.

-Then it creates a scatter plot of the original, non-standardized data for the independent variable (X) and the dependent variable (y)

-It uses the inverse_transform method from the StandardScaler objects sc_X and sc_y to convert the data back to its original scale.
"""

X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Position Salary Prediction (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
