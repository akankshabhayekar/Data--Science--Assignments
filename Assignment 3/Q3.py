# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:02:41 2022

@author: akank
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#Reding the Desired Data File

df = pd.read_csv(r"C:\Users\akank\Documents\IC 142\pima-indians-diabetes.csv")

for i in df :
    
    plt.show()
    M = np.median(df[i])
    q1 = df[i].quantile(0.25)
    q2 = df[i].quantile(0.75)
    IQR = q2 - q1
    Q1 = q1 - 1.5*IQR
    Q2 = q2 + 1.5*IQR
    
    for j in range(len(df[i])) :
        if (df[i][j]<Q1 or df[i][j]>Q2):
            
            df.loc[j,i] = M
    
    
df1=df.copy()
df2=df.copy()
for i in df2 :
    m = df2[i].mean()
    s = df2[i].std()
    for j in range(len(df2[i])) :
        df2.loc[j,i] = (df2[i][j] - m )/s

# Q 3

df2 = df2.iloc[:,:7]
for i in df2 :
    m = df2[i].mean()
    for j in range(len(df2[i])) :
        df2.loc[j,i] = (df2[i][j] - m )

# PART A

co1 = np.dot(np.transpose(df2),df2)
w,v=np.linalg.eig(co1)

w1 = list(w)
w1.sort(reverse=True)

# Reducing the Dimension to 2

df3 = np.dot(df2,v[:,0:2])

co2 = np.dot(np.transpose(df3),df3)

# Scatterplot of the Reduced Data

plt.scatter(df3[:,0],df3[:,1], color= "violet")
plt.title('Scatterplot for the Transformed Data',size=22)
plt.grid(color='grey', linestyle='-.', linewidth=0.7)
plt.show()

# PART B

# Plotting Eigenvalues in the Descending Order

l=[1,2,3,4,5,6,7]
plt.plot(l,w1,marker='o', color='red')
plt.title('Eigenvalues in the Descending Order',size=22, color='black')
plt.grid(color='grey', linestyle='-.', linewidth=0.7)
plt.show()

# PART C

# Plotting the Reconstruction Error w.r.t l

from numpy import linalg as LA
from sklearn.decomposition import PCA

error_record=[]
for i in range(1,8):
    pca = PCA(n_components=i, random_state=50)
    pca2_results = pca.fit_transform(df2)
    pca2_proj_back=pca.inverse_transform(pca2_results)
    total_loss=LA.norm((df2-pca2_proj_back),None)
    error_record.append(total_loss)
l2 = [i for i in range(1,8)]

plt.title("Reconstruct Error of PCA",size=22)
plt.plot(l2,error_record,'r',marker='o', color='blue')
plt.xlabel('No of Dimension (l)')
plt.ylabel('Euclidean Distance')
plt.grid(color='grey', linestyle='-.', linewidth=0.7)
plt.show()

# PART D

df4 = np.dot(df2,v)

co3 = np.dot(np.transpose(df4),df4)
print(co1)
print(co3)