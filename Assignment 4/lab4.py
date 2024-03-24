# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:25:13 2022

@author: akank
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split

 
#Question 1
df = pd.read_csv(r"C:\Users\akank\Documents\252 LAB4\SteelPlateFaults-2class.csv")
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, random_state=42)


x_train.to_csv("SteelPlateFaults-train",index=True)

x_test.to_csv("SteelPlateFaults-test",index=True)

a=[1,3,5]
#confusion matrix for k = 1,3 and 5
for i in a:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print("The confusion matrix for k=",i, " is ", cm)
    plt.figure(figsize=(7,5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    a1 = accuracy_score(y_test, y_pred)
    print("The accuracy score for value k=",i," is  ",a1)


#Question 2
df1=pd.read_csv(r"C:\Users\akank\Desktop\SteelPlateFaults-train")
df3=df
df2=pd.read_csv(r"C:\Users\akank\Desktop\SteelPlateFaults-test")
df4=df
for i in df1 :
    m_i_n = df1[i].min()
    m_a_x = df1[i].max()
    for j in range(len(df1[i])) :
        df1.loc[j,i] = (df1[i][j] - m_i_n )/(m_a_x-m_i_n)
for i in df2 :
    
    for j in range(len(df2[i])) :
        df2.loc[j,i] = (df2[i][j] - m_i_n )/(m_a_x-m_i_n)

df1.to_csv("SteelPlateFaults-train-normalised",index=True)

df2.to_csv("SteelPlateFaults-test-normalised",index=True)

#confusion matrix for k = 1,3 and 5
for i in a:
    
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(df1,y_train)
    y_pred = model.predict(df2)
    cm = confusion_matrix(y_test, y_pred)
    print("The confusion matrix for k=",i, " is ", cm)
    plt.figure(figsize=(7,5))
    sn.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    a1 = accuracy_score(y_test, y_pred)
    print("The accuracy score for value k=",i ," is  ",a1)
    
#Question 3
model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("The confusion matrix for bayes classifier is ", cm)
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
a1 = accuracy_score(y_test, y_pred)
print("The accuracy score for bayes classifier is ", a1)










