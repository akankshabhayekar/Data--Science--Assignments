# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:35:44 2022

@author: akank
"""
#Nmae=Akanksha Bhayekar
#Roll.no=B21148
#Mobile no=8177936098

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import math
from statsmodels.tsa.ar_model import AutoReg
sns.set_theme(style="whitegrid")

# Q2(A) Splitting the data into Training and Testing

print("Q2(a)-")
data = pd.read_csv(r"C:\Users\akank\Documents\ic160\daily_covid_cases.csv",index_col=0,parse_dates=True)
X=data.values

test_size=0.35

train,test=X[:len(X)-math.ceil(test_size*len(X))],X[len(X)-math.ceil(test_size*len(X)):]

# Plotting the Train and Test set

print("The Plot of Training Data is shown above.")
plt.plot(train, color='blue')
plt.grid(color='black')
plt.title("Plot of the Training Data")
plt.show()
print("The Plot of Training Data is shown above.")
plt.plot(test, color='green')
plt.grid(color='black')
plt.title("Plot of the Testing Data")
plt.show()

# Modelling the Auto-Regeression
 
model = AutoReg(train,lags=5 ,old_names=False)
model_fit = model.fit() # Fit/Train the model
coef = model_fit.params # Get the Coefficients of AR Model

print("Coefficients (w0,w1,...,w5) from the Trained Auto-Regression Model are as follows: ")
for i in range(5):
    print("w",i,":",round(coef[i],3))
    
#Q2(B)
print("\nQ2(b)-")

# Using these coefficients walk forward over time steps in test, one step each time

history = train[len(train)-5:]
history = [history[i] for i in range(len(history))]

predictions = [] # List to hold the predictions, 1 step at a time

for  t in range(len(test)):
    length=len(history)
    lag =  [history[i] for i in range(length-5,length)]
    yhat = coef[0] # Intialize to w0
    for d in  range(5):
        yhat+=coef[d+1]*lag[5 - d-1]    # Add other values
    obs = test[t] 
    predictions.append(yhat[0]) # Append predictions to compute RMSE later 
    history.append(obs) # Append actual test value to history that to be used in next step
            
# Scatter Plot between Actual and Predicted Test Values 
 
print("Q2(b-i)- The Plot is shown above.")
predictions=pd.DataFrame(predictions)
test=pd.DataFrame(test)
sns.scatterplot(x=test[0],y=predictions[0],color="orange")
plt.title("Scatter Plot") 
plt.xticks(rotation=270)
plt.xlabel("Actual Tested Values")
plt.ylabel("Predicted Tested Values")
plt.show()

# Line Plot between the Actual and the Predicted Test Data 

print("Q2(b-ii)-The Plot is shown above.")
plt.plot(test[0],color="red")
plt.plot(predictions[0],color="black")
plt.title("Line Plot")
plt.grid(color='black')
plt.legend(["Predicted", "Actual"])
plt.show()

# RMSE(%) and MAPE between the Actual and the Predicted Test Data

rmse =(math.sqrt(mean_squared_error(test,predictions))/(np.mean(test)))*100
mape=mean_absolute_percentage_error(test,predictions)*100
print("RMSE(%) between the Actual and the Predicted Test Data is: ",round(rmse[0],3))
print("MAPE between the Actual and the Predicted Test Data is: ",round(mape,3))
