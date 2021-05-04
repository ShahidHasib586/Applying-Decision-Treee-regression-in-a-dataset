 #importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the dataset

dataset = pd.read_csv('Position_Salaries.csv')

#putting indepandent variabls value

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values
##Y = Y.reshape(len(Y),1)

#fitting the data set into decision tree regressor

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)
#predict new result
Y_pred = regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 