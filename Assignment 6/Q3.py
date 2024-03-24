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
print("Q3-")
data = pd.read_csv(r"C:\Users\akank\Documents\ic160\daily_covid_cases.csv",index_col=0,parse_dates=True)
X=data.values

test_size=0.35

train,test=X[:len(X)-math.ceil(test_size*len(X))],X[len(X)-math.ceil(test_size*len(X)):]

# Modelling the Auto-Regression

RMSE=[]
MAPE=[] 
def model_AR(windows):
        model = AutoReg(train,lags=windows ,old_names=False)
        model_fit = model.fit() #fit/train the model
        coef = model_fit.params #Get the coefficients of AR model
        
        # Using these Coefficients walk forward over time steps in test, one step each time
        
        history = train[len(train)-windows:]
        history = [history[i] for i in range(len(history))]
        
        predictions = [] # List to hold the predictions, 1 step at a time
        
        for  t in range(len(test)):
            length=len(history)
            lag =  [history[i] for i in range(length-windows,length)]
            yhat = coef[0] # Intialize to w0
            for d in  range(windows):
                yhat+=coef[d+1]*lag[windows - d-1]  # Add other values
            obs = test[t] 
            predictions.append(yhat[0]) # Append predictions to compute RMSE later 
            history.append(obs)  # Append actual test value to history, to be used in next step
                    
        
        
        # RMSE(%) and MAPE between the Actual and the Predicted Test Data
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_percentage_error
        
        rmse =(math.sqrt(mean_squared_error(test,predictions))/(np.mean(test)))*100
        mape=mean_absolute_percentage_error(test,predictions)*100
        print("For lag-",windows,"RMSE(%) between actual and predicted test data: ",round(rmse,3))
        print("For lag-",windows,"MAPE between actual and predicted test data: ",round(mape,3))
        print("\n")
        RMSE.append(round(rmse,3))
        MAPE.append(round(mape,3))

lags=[1,5,10,15,25]
for i in lags:
    model_AR(i)
lags=pd.DataFrame(lags)
RMSE=pd.DataFrame(RMSE)
MAPE=pd.DataFrame(MAPE)
sns.barplot(x=lags[0], y=RMSE[0])
plt.ylabel("RMSE (%)")
plt.xlabel("Lagged-Values")
plt.show()

sns.barplot(x=lags[0], y=MAPE[0])
plt.ylabel("MAPE")
plt.xlabel("Lagged-Values")
plt.show()
