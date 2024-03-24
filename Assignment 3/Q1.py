# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 21:55:43 2022

@author: akank
"""


import pandas as pd
import numpy as np


# Reading the Desired CSV File
df=pd.read_csv(r"C:\Users\akank\Documents\IC 142\pima-indians-diabetes.csv")

#QUESTION 1.(a)

for i in df :
    
    med_ian = np.median(df[i])
    q1 = df[i].quantile(0.25)
    q2 = df[i].quantile(0.75)
    IQR = q2 - q1
    
    Q1 = q1 - 1.5*IQR
    Q2 = q2 + 1.5*IQR
    
    for j in range(len(df[i])) :
        if (df[i][j]<Q1 or df[i][j]>Q2):
            
            df.loc[j,i] = med_ian
    
    
df1=df.copy()
df2=df.copy()
print('Minimum Before:')
print(df1.min())
print()
print('Maximum before:')
print(df1.max())
print()
for i in df1 :
    m_i_n = df1[i].min()
    m_a_x = df1[i].max()
    for j in range(len(df1[i])) :
        df1.loc[j,i] = (df1[i][j] - m_i_n )*(7)/(m_a_x-m_i_n) + 5
print('Minimum After:')
print(df1.min())
print()
print('Maximum After:')
print(df1.max())
print()

#QUESTION 1.(b)
print('Mean Before:')
print(df2.mean())
print()
print('STD Before:')
print(df2.mean())
print()
for i in df2 :
    m_e_a_n = df2[i].mean()
    s_t_d = df2[i].std()
    for j in range(len(df2[i])) :
        df2.loc[j,i] = (df2[i][j] - m_e_a_n )/(s_t_d)

print('Mean After:')
print(df2.mean())
print()
print('STD After:')
print(df2.std())