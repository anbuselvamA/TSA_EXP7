# Developer name :A.Anbuselvam
# Reg no : 212222240009
# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the student performance dataset, convert it to a DataFrame, and set the StudentID as the index to treat it as the time series variable.
3. Visualize the FinalGrade over the student IDs to see the time series trend.
4.Fit an AR model using the past 2 lags (p=2), which means predicting each value based on the two previous grades.
5. Use the trained AR model to predict the next 5 student grades)
6. Plot the original grades and the predicted values from the AR model.Apply the Holt-Winters
7. Exponential Smoothing method with an additive trend (no seasonality).
8. Predict the next 5 student grades using the ES model.
9. Plot the original grades, the fitted values, and the forecasted grades from the ES model.

### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Create the student performance dataset
df = pd.read_csv('/content/student_performance.csv')

# Convert to DataFrame
df = pd.DataFrame(data)

# Treat StudentID as the time index for simplicity
df.set_index('StudentID', inplace=True)

# Plot the data
plt.plot(df.index, df['FinalGrade'], marker='o')
plt.title('Final Grades Over Student IDs')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.grid(True)
plt.show()

# ---------------------------
# Auto-Regressive (AR) Model
# ---------------------------
# Fit AR model (p=2, using 2 lag values for prediction)
ar_model = AutoReg(df['FinalGrade'], lags=2).fit()

# Predict the next 5 final grades using the AR model
ar_predictions = ar_model.predict(start=len(df), end=len(df) + 4)

print("AR Model Predictions for next 5 students:\n", ar_predictions)

# Plot AR results
plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')
plt.plot(range(len(df) + 1, len(df) + 6), ar_predictions, label='AR Forecasted Grades', marker='x')
plt.title('Auto-Regressive Model Forecast')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------
# Exponential Smoothing (ES)
# ---------------------------
# Apply Holt-Winters Exponential Smoothing (Additive trend, no seasonality)
es_model = ExponentialSmoothing(df['FinalGrade'], trend='add', seasonal=None).fit()

# Predict the next 5 final grades using the ES model
es_predictions = es_model.forecast(steps=5)

print("Exponential Smoothing Predictions for next 5 students:\n", es_predictions)

# Plot ES results
plt.plot(df.index, df['FinalGrade'], label='Original Final Grades', marker='o')
plt.plot(df.index, es_model.fittedvalues, label='ES Fitted Values', linestyle='--')
plt.plot(range(len(df) + 1, len(df) + 6), es_predictions, label='ES Forecasted Grades', marker='x')
plt.title('Exponential Smoothing Forecast')
plt.xlabel('Student ID')
plt.ylabel('Final Grade')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:
![Untitled-1](https://github.com/user-attachments/assets/2ee7ecac-ab33-405d-97af-eb60acb3c339)

# AR Model Predictions for next 5 students:
```
 10    75.561912
11    81.415614
12    78.162836
13    79.863026
14    78.954766
dtype: float64
```
![Untitled](https://github.com/user-attachments/assets/d52b436b-153d-4c20-91df-1c5b1839c4d1)
# Exponential Smoothing Predictions for next 5 students:
```
 10    77.933743
11    77.521729
12    77.109715
13    76.697701
14    76.285687
dtype: float64
```
![Untitled](https://github.com/user-attachments/assets/2688aaa5-45d5-4910-8442-2aa40167e69b)




### RESULT:
Thus the have successfully implemented the auto regression function using python.
