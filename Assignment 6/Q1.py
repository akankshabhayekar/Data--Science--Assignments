# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 19:35:45 2022

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

# Importing the Desired CSV File.
data = pd.read_csv(r"C:\Users\akank\Documents\ic160\daily_covid_cases.csv")


# (1) Autocorrelation line plot with lagged values

# Q1(A) Line Plot of the Data
# Generating the X-Ticks
x = [16]
for i in range(10):
    x.append(x[i] + 60)

list=["Feb-20","Apr-20","Jun-20","Aug-20","Oct-20","Dec-20","Feb-21","Apr-21","Jun-21","Aug-21","Oct-21"]    
print("Q1(a)-") 
print("The grpah is shown above.")   
data.plot(color="red")
plt.xticks(x,list,rotation=270)
plt.ylabel("New COVID-Confirmed Cases")
plt.xlabel("Month and Year")
plt.show()

# Q1(B) Autocorrelation

print("\nQ1(b)-")
data_lag = pd.Series(data.iloc[0:611]["new_cases"])  # Lag-Time Series 

data_t = pd.Series(data.iloc[1:612]["new_cases"])  # Given Time Sequence 
data_t.index=[i for i in range(611)]

print("Autocorrelation between 1-day Lag and the given Time Sequence:",round(data_t.corr(data_lag),3))

# Q1(C) Scatterplot

print("\nQ1(c)-")
print("The grpah is shown above.")
sns.scatterplot(x=data_lag,y=data_t,color="violet")
plt.ylabel("Data-at t")
plt.xlabel("Data at t-1")
plt.xticks(rotation=270)
plt.show()

# Q1(D)
 
print("\nQ1(d)-")
auto_corr=[]
for i in range(1,7):
    data_lag = pd.Series(data.iloc[0:612-i]["new_cases"]) ##lag-time series 
    data_t = pd.Series(data.iloc[i:612]["new_cases"])  ##given time squence 
    data_t.index=[i for i in range(612-i)]
    
    print("Autocorrelation between ",i,"day-lag and the given Time Sequence:",round(data_t.corr(data_lag),3))
    auto_corr.append(round(data_t.corr(data_lag),3))

plt.plot([i for i in range(1,7)],auto_corr, color='blue')
plt.xlabel("Lagged Values")
plt.ylabel("Correlation Coefficients")
plt.show()

# Q1(E) Plot a Correlogram or Auto Correlation Function

print("\nQ1(e)-")
print("The grpah is shown above.")
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data["new_cases"])
plt.xlabel("Lagged Values")
plt.ylabel("Correlation Coefficients")
plt.show()
