import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score


# Importing data
dataset = pd.read_csv('Bias_correction_ucl.csv')
X = dataset.iloc[:-2, :-2].values
y_max = dataset.iloc[:-2, -2].values
y_min = dataset.iloc[:-2, -1].values

# Reshape for scaling later on
y_max = y_max.reshape(len(y_max), 1)
y_min = y_min.reshape(len(y_min), 1)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, 2:])

X[:, 2:] = imputer.transform(X[:, 2:])

# Do the same for dependent variables
imputer_y = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_y.fit(y_min)
imputer_y.fit(y_max)

# Do the replacing
y_min = imputer_y.transform(y_min)
y_max = imputer_y.transform(y_max)

# Encoding categorical data
from datetime import datetime as dt
for sublist in X:
    print("Date as string saved: {}".format(sublist[1]))
    d = dt.strptime(sublist[1], '%Y-%m-%d').date()
    print("Date with new format as number: {}, of type {}".format(d.toordinal(), type(d.toordinal())))
    sublist[1] = d.toordinal()

# Splitting the data
# I will do 2 splits because we have 2 variables to p[redict]
X_train, X_test, y_train_min, y_test_min = train_test_split(X, y_min, test_size=0.2, random_state=1)
X_train, X_test, y_train_max, y_test_max = train_test_split(X, y_max, test_size=0.2, random_state=1)

# Feature scaling
sc_X = StandardScaler()
sc_y_min = StandardScaler()
sc_y_max = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

y_train_min = sc_y_min.fit_transform(y_train_min)
y_test_min = sc_y_min.transform(y_test_min)

y_train_max = sc_y_max.fit_transform(y_train_max)
y_test_max = sc_y_max.transform(y_test_max

# Train the SVR model for minimum temperatures
regressor_min = SVR(kernel = 'rbf') # rfb = radial basis function (non linear function)
regressor_min.fit(X_train, y_train_min)

# Train the SVR for max temps
regressor_max = SVR(kernel = 'rbf') # rfb = radial basis function (non linear function)
regressor_max.fit(X_train, y_train_max)

# Predicting test results for min temps
y_pred_min = sc_y_min.inverse_transform(regressor_min.predict(X_test))

# ...and for max temps
y_pred_max = sc_y_max.inverse_transform(regressor_max.predict(X_test))

# Evaluating performance for min temps
print(r2_score(sc_y_min.inverse_transform(y_test_min), y_pred_min))

for i in range(len(y_pred_min)):
    print("Predicted outcome: {}/ Actual outcome: {}".format(y_pred_min[i], sc_y_min.inverse_transform(y_test_min[i])))

# ...and for max temps
print(r2_score(sc_y_max.inverse_transform(y_test_max), y_pred_max))

for i in range(len(y_pred_max)):
	print("Predicted outcome: {}/ Actual outcome: {}".format(y_pred_max[i], sc_y_max.inverse_transform(y_test_max[i])))