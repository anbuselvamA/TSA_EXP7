# Developer name :A.Anbuselvam
# Reg no : 212222240009
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the student performance dataset, convert it to a DataFrame, and set the StudentID as the index to treat it as the time series variable.
3. You will need libraries like: pandas for data manipulation statsmodels for AR model statsmodels.tsa.holtwinters for Exponential Smoothing
4.The dataset is first split into training and testing sets. The non-numeric columns like StudentID, Name, and Gender are dropped since we're focusing on numeric features.
5. We use the AutoReg model from statsmodels with a lag of 2 (you can experiment with the lag parameter). The AR model uses previous values of FinalGrade to predict future values
6.  Predictions are then plotted alongside the actual values of FinalGrade for comparison.
### PROGRAM
```
 # Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/student_performance(1).csv')



# Prepare the data for modeling (Use 'FinalGrade' as the target variable)
target = 'FinalGrade'

# Step 1: Split data into training and testing sets
X = df.drop(columns=[target, "StudentID", "Name", "Gender"])  # Drop non-numeric columns
y = df[target]

# Since we only have 10 data points, let's use 80% training data and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ---------------------------------------
# Step 2: AutoRegressive Model (AR model)
# ---------------------------------------
# Train the AutoRegressive model
model_ar = AutoReg(y_train, lags=2)  # We choose lag=2 for this example
model_ar_fitted = model_ar.fit()

# Make predictions
y_pred_ar = model_ar_fitted.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# Evaluate AR model (using RMSE)
from sklearn.metrics import mean_squared_error
import math

rmse_ar = math.sqrt(mean_squared_error(y_test, y_pred_ar))
print(f"RMSE for AutoRegressive model: {rmse_ar}")

# Plot AR predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual FinalGrade", color="blue")
plt.plot(y_test.index, y_pred_ar, label="Predicted FinalGrade", color="red", linestyle="--")
plt.title("AutoRegressive Model: FinalGrade Predictions")
plt.xlabel("Student Index")
plt.ylabel("FinalGrade")
plt.legend()
plt.show()

# ---------------------------------------
# Step 3: Exponential Smoothing (ETS)
# ---------------------------------------
# Applying Holt-Winters Exponential Smoothing Model
model_es = ExponentialSmoothing(y_train, trend='add', seasonal=None, seasonal_periods=None)
model_es_fitted = model_es.fit()

# Make predictions
y_pred_es = model_es_fitted.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# Evaluate Exponential Smoothing model (using RMSE)
rmse_es = math.sqrt(mean_squared_error(y_test, y_pred_es))
print(f"RMSE for Exponential Smoothing model: {rmse_es}")

# Plot Exponential Smoothing predictions vs actual
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual FinalGrade", color="blue")
plt.plot(y_test.index, y_pred_es, label="Predicted FinalGrade", color="green", linestyle="--")
plt.title("Exponential Smoothing Model: FinalGrade Predictions")
plt.xlabel("Student Index")
plt.ylabel("FinalGrade")
plt.legend()
plt.show()


```
### OUTPUT:
RMSE for AutoRegressive model: 11.05098267829926
![Untitled](https://github.com/user-attachments/assets/4260702f-7e60-4e1a-b277-43476a183aef)

RMSE for Exponential Smoothing model: 9.602243537611528
![Untitled-1](https://github.com/user-attachments/assets/f27222c1-ba91-4478-87d9-dfa117f92890)
### RESULT:
Thus the have successfully implemented the auto regression function using python.
