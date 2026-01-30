# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import reqired libraries and create the datasets with study hours and marks
2. divide the data sets into training and training sets
3. use the trained modelto predict the models on testing the data and predicted the output


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Dhakshinkumar C
RegisterNumber:  212225240031
*/

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([10, 20, 30, 50, 52]).reshape(-1, 1)
y = np.array([12, 14, 16, 18, 20])

model = LinearRegression()
model.fit(x, y)
y_ = model.predict(x)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(x, y, color='purple', label="actual_Data")
plt.plot(x, y_, color='green', label="Regression_Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```
## Output:
<img width="717" height="547" alt="image" src="https://github.com/user-attachments/assets/7b5dc2ee-96c8-4473-9849-34884420c75b" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
