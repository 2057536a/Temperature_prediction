import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# IMPORTING THE DATA
dataset = pd.read_csv('Bias_correction_ucl.csv')
X = dataset.iloc[:-2, :-2].values
y_max = dataset.iloc[:-2, -2].values
y_min = dataset.iloc[:-2, -1].values

# TAKING CARE OF MISSING DATA
dataset.describe()

# Use the Imputer to replace all nan values in columns with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Apply it to X - to all its rows and for columns 2 
# and 3 because column 1 is categorical
imputer.fit(X[:, 2:])

# Do the replacing
X[:, 2:] = imputer.transform(X[:, 2:])

# Use the Imputer to replace all nan values in columns with mean
imputer_y = SimpleImputer(missing_values=np.nan, strategy='mean')

# Apply it to X - to all its rows and for columns 2 
# and 3 because column 1 is categorical
imputer_y.fit(y_min.reshape(-1, 1))
imputer_y.fit(y_max.reshape(-1, 1))

# Do the replacing
y_min = imputer_y.transform(y_min.reshape(-1, 1))
y_max = imputer_y.transform(y_max.reshape(-1, 1))

# ENCODING CATEGORICAL DATA
from datetime import datetime as dt
for sublist in X:
    print("Date as string saved: {}".format(sublist[1]))
    d = dt.strptime(sublist[1], '%Y-%m-%d').date()
    print("Date with new format as number: {}, of type {}".format(d.toordinal(), type(d.toordinal())))
    sublist[1] = d.toordinal()

# SPLITTING DATA
X_train, X_test, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=1)
X_train, X_test, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=1)

# TRAINING THE MULTUPLE REGRESSION MODEL FOR Y MIN
regressor_min = LinearRegression()
regressor_min.fit(X_train, y_train_min)

# TRAINING THE MULTUPLE REGRESSION MODEL FOR Y MAX
regressor_max = LinearRegression()
regressor_max.fit(X_train, y_train_max)

# PREDICTING TEST SET RESULTS FOR Y MIN
y_min_pred = regressor_min.predict(X_test)

# PREDICTING TEST SET RESULTS FOR Y MAX
y_max_pred = regressor_max.predict(X_test)

# EVALUATING RESULTS FOR Y MIN
print(regressor_min.coef_)
print(regressor_min.intercept_)
print(r2_score(y_test_min, y_min_pred))

for i in range(len(y_min_pred)):
    print("Predicted: {}/ Actual: {}".format(y_min_pred[i], y_test_min[i]))

# EVALUATING RESULTS FOR Y MAX
print(regressor_max.coef_)
print(regressor_max.intercept_)
print(r2_score(y_test_max, y_max_pred))

for i in range(len(y_max_pred)):
    print("Predicted: {}/ Actual: {}".format(y_max_pred[i], y_test_max[i]))

