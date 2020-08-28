import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from datetime import datetime as dt
from sklearn.model_selection import train_test_split

# Importing the data
dataset = pd.read_csv('Bias_correction_ucl.csv')
X = dataset.iloc[:-2, :-2].values
y_max = dataset.iloc[:-2, -2].values
y_min = dataset.iloc[:-2, -1].values

y_max = y_max.reshape(len(y_max), 1)
y_min = y_min.reshape(len(y_min), 1)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:])
X[:, 2:] = imputer.transform(X[:, 2:])

# Missing data for dependent variables
imputer_y = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_y.fit(y_min)
imputer_y.fit(y_max)
y_min = imputer_y.transform(y_min)
y_max = imputer_y.transform(y_max)

# Encoding categorical data
for sublist in X:
    print("Date as string saved: {}".format(sublist[1]))
    d = dt.strptime(sublist[1], '%Y-%m-%d').date()
    print("Date with new format as number: {}, of type {}".format(d.toordinal(), type(d.toordinal())))
    sublist[1] = d.toordinal()

# Splitting data
X_train, X_test, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=1)
X_train, X_test, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=1)

# Train the random forrest regression for minimums
regressor_min = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_min.fit(X_train, y_train_min)

# Predicting test results for min temps
y_pred_min = regressor_min.predict(X_test)

# Evaluating the RFR model for min temps
for i in range(len(y_pred_min)):
    print("Predicted outcome: {}/ Actual outcome: {}".format(y_pred_min[i], y_test_min[i]))
print(r2_score(y_test_min, y_pred_min))

# Train the random forrest regression for maximums
regressor_max = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor_max.fit(X_train, y_train_max)

# Predicting test results for min temps
y_pred_max = regressor_max.predict(X_test)

# Evaluating the RFR model for min temps
for i in range(len(y_pred_max)):
    print("Predicted outcome: {}/ Actual outcome: {}".format(y_pred_max[i], y_test_max[i]))
print(r2_score(y_test_max, y_pred_max))