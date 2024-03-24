# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:38:15 2022

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
import math
from statsmodels.tsa.ar_model import AutoReg
sns.set_theme(style="whitegrid")

print("Q4-")

data = pd.read_csv(r"C:\Users\akank\Documents\ic160\daily_covid_cases.csv",index_col=0,parse_dates=True)
X=data.values

test_size=0.35

train,test=X[:len(X)-math.ceil(test_size*len(X))],X[len(X)-math.ceil(test_size*len(X)):]


def model_AR(windows):
        model = AutoReg(train,lags=windows ,old_names=False)
        model_fit = model.fit() # Fit/Train the model
        coef = model_fit.params # Get the Coefficients of AR Model
        
        # Using these coefficients walk forward over time steps in test, one step each time
        history = train[len(train)-windows:]
        history = [history[i] for i in range(len(history))]
        
        predictions = [] # List to hold the predictions, 1 step at a time
        
        for  t in range(len(test)):
            length=len(history)
            lag =  [history[i] for i in range(length-windows,length)]
            yhat = coef[0] #Intialize to w0
            for d in  range(windows):
                yhat+=coef[d+1]*lag[windows - d-1]  # Add other values
            obs = test[t] 
            predictions.append(yhat[0]) # Append predictions to compute RMSE later 
            history.append(obs) # Append actual test value to history, to be used in next step
                    
        
        
        # RMSE(%) and MAPE between actual and predicted test data
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_percentage_error
        
        rmse =(math.sqrt(mean_squared_error(test,predictions))/(np.mean(test)))*100
        mape=mean_absolute_percentage_error(test,predictions)*100
        print("For lag-",windows,"RMSE(%) between the Actual and the Predicted Test Data is: ",round(rmse,3))
        print("For lag-",windows,"MAPE between the Actual and the Predicted Test Data is: ",round(mape,3))

heuristic_value=0
alpha=pd.DataFrame(train)
for lags in range(1,397):
    data_lag = pd.Series(alpha.iloc[0:397-lags][0]) # Lag-Time Series 
    
    data_t = pd.Series(alpha.iloc[lags:397][0])  # Given Time Sequence 
    data_t.index=[i for i in range(397-lags)]
    ar=data_t.corr(data_lag)
    if(np.absolute(ar)<2/np.sqrt(397-lags)):
        break
    else:
        heuristic_value=lags
print("Heuristic Value is: ",heuristic_value)

model_AR(heuristic_value) # Calling the Function

