import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# IMPORTING THE DATA
dataset = pd.read_csv('Bias_correction_ucl.csv')
X = dataset.iloc[:-2, :-2].values
y_max = dataset.iloc[:-2, -2].values
y_min = dataset.iloc[:-2, -1].values

# TAKING CARE OF MISSING VALUES
# Use the Imputer to replace all nan values in columns with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Apply it to X - to all its rows and for columns 2 
# and 3 because column 1 is categorical
imputer.fit(X[:, 2:])

# Do the replacing
X[:, 2:] = imputer.transform(X[:, 2:])

# MISSING DATA ALSO IN DEPENDENT VARIABLES
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
for sublist in X:
    print("Date as string saved: {}".format(sublist[1]))
    d = dt.strptime(sublist[1], '%Y-%m-%d').date()
    print("Date with new format as number: {}, of type {}".format(d.toordinal(), type(d.toordinal())))
    sublist[1] = d.toordinal()

# SPLITTING DATA FOR Y MIN AND MAX VARIABLES SEPARATELY
X_train, X_test, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=1)
X_train, X_test, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=1)

# TRAINING THE POLYNOMIAL REGRESSION FOR MINIMUM TEMPERATURES
poly_reg_min = PolynomialFeatures(degree = 4)
X_poly_min = poly_reg_min.fit_transform(X_train)
regressor_min = LinearRegression()
regressor_min.fit(X_poly_min, y_train_min)

# TRAINING THE POLYNOMIAL REGRESSION FOR MAXIMUM TEMPERATURES
poly_reg_max = PolynomialFeatures(degree = 4)
X_poly_max = poly_reg_max.fit_transform(X_train)
regressor_max = LinearRegression()
regressor_max.fit(X_poly_max, y_train_max)

# PREDICTING RESULTS FOR MINIMUM TEMPERATURES
y_pred_min = regressor_min.predict(poly_reg_min.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_min.reshape(len(y_pred_min),1), y_test_min.reshape(len(y_test_min),1)),1))

# PREDICTING RESULTS FOR MAXIMUM TEMPERATURES
y_pred_max = regressor_max.predict(poly_reg_max.transform(X_test))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred_max.reshape(len(y_pred_max),1), y_test_mmax.reshape(len(y_test_max),1)),1))

# EVALUATING MIN TEMPERATURES PREDICTION
print(r2_score(y_test_min, y_pred_min))

# EVALUATING MAX TEMPERATURES PREDICTION
print(r2_score(y_test_max, y_pred_max))