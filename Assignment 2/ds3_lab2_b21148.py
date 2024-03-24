#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 20:13:19 2022

@author: deepakjangid
"""

import pandas as pd 
import math
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\akank\Documents\252 LAB4\landslide_data3_miss.csv")
mainDf = df
df2 = pd.read_csv(r"C:\Users\akank\Documents\252 LAB4\landslide_data3_original.csv")
col = ['stationid','temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']

# Question 1

missVal = []
i = col[0]
for i in col:
    count = df[i].isnull().sum()
    missVal.append(count)

plt.bar(col,missVal)
plt.xticks(rotation = 315)
plt.ylabel('Number of missing values')
plt.show()

# Question 2
print("<-----Question 2(a)-------->")
print()
# a
count = df["stationid"].isnull().sum()
df = df.dropna(subset = ['stationid']) # dropping the rows having null vlues in stationid
print("Number of dropped rows in stationid is "+str(count))

print()
print("<-----Question 2(b)-------->")
print()
# b
df = df.transpose()
droppedRows = 0
for i in df:
    count = 0
    for j in df[i]:
        if(pd.isna(j)):
            count += 1
    if(count>=3):
        df.drop(i,inplace = True,axis = 1)
        droppedRows += 1

df = df.transpose()
print("Number of dropped rows is "+str(droppedRows))
print()

# Question 3
print("<-----Question 3-------->")
print()

totalMissingVal = 0
for i in col:
    x = df[i].isnull().sum()
    print("Number of Missing value in ",i," is ",x)
    totalMissingVal += x
print()
print("Total number of missing values in df is ",totalMissingVal)
print()

print("<-----Question 4-------->")
print()

df = mainDf

# Question 4 a i
print("<---------Question 4a--------->")
print()
for i in col:
    if(i!="stationid"):
        mean = df[i].mean()
        df[i].fillna(mean,inplace = True)

for i in col:
    if(i!="stationid"):
        mean1 = df[i].mean()
        mean2 = df2[i].mean()
        mode1 = float(df[i].mode())
        mode2 = float(df2[i].mode())
        median1 = df[i].median()
        median2 = df2[i].median()
        sd1 = df[i].std()
        sd2 = df2[i].std()
        print("for",i)
        print("calculated value of mean is",mean1,"and for original data mean is",mean2)
        print("calculated value of median is",median1,"and for original data median is",median2)
        print("calculated value of mode is",mode1,"and for original data mode is",mode2)
        print("calculated value of sd is",sd1,"and for original data sd is",sd2)
        print()
    
#Question 4 a ii 
arr = [0]

for i in col:
    sum = 0
    n = 0 
    if(i=="stationid"):
        continue
    for j in range(len(df[i])):
        x = (df[i][j] - df2[i][j])**2
        if(x!=0):
            n += 1
            sum += x
    arr.append(math.sqrt(sum/n))

plt.bar(col,arr)
plt.xticks(rotation = 315)
plt.show()

print("<---------Question 4b--------->")
print()

# Question 4 b i
df = mainDf
for i in col:
    if(i!="stationid"):
        x = df[i].interpolate()
        df[i].fillna(x,inplace = True)
for i in col:
    if(i!="stationid"):
        mean1 = df[i].mean()
        mean2 = df2[i].mean()
        mode1 = float(df[i].mode())
        mode2 = float(df2[i].mode())
        median1 = df[i].median()
        median2 = df2[i].median()
        sd1 = df[i].std()
        sd2 = df2[i].std()
        print("for",i)
        print("calculated value of mean is",mean1,"and for original data mean is",mean2)
        print("calculated value of median is",median1,"and for original data median is",median2)
        print("calculated value of mode is",mode1,"and for original data mode is",mode2)
        print("calculated value of sd is",sd1,"and for original data sd is",sd2)
        print()

# Question 4b ii
arr = [0]

for i in col:
    sum = 0
    n = 0 
    if(i=="stationid"):
        continue
    for j in range(len(df[i])):
        x = (df[i][j] - df2[i][j])**2
        if(x!=0):
            n += 1
            sum += x
    arr.append(math.sqrt(sum/n))

plt.bar(col,arr)
plt.xticks(rotation = 315)
plt.show()

print("<---------Question 5 a---------->")
print()
Q1 = df["temperature"].quantile(0.25)
Q3 = df["temperature"].quantile(0.75)
IQR = Q3-Q1

outLiers1 = []
for i in df["temperature"]:
    if(i<(Q1 - 1.5*IQR) or i>(Q3 + 1.5*IQR)):
        outLiers1.append(i)
print("OutLiers in temerature are",outLiers1)
print()
q1 = df["rain"].quantile(0.25)
q3 = df["rain"].quantile(0.75)
IqR = q3-q1

outLiers2 = []
for i in df["rain"]:
    if(i<(q1 - 1.5*IqR) or i>(q3 + 1.5*IqR)):
        outLiers2.append(i)

print("OutLiers in rain are",outLiers2)

# printing Box plot for temperature and rain
plt.boxplot(df["temperature"])
plt.title("Box Plot for temperature")
plt.show()
plt.boxplot(df["rain"])
plt.title("Box Plot for rain")
plt.show()


# Question 5 b
print("<---------Question 5b--------->")

med1 = df["temperature"].median()
for i in df["temperature"]:
    if(i<(Q1 - 1.5*IQR) or i>(Q3 + 1.5*IQR)):
        df["temperature"].replace(i,med1,regex = True)
plt.boxplot(df["temperature"])
plt.title("Box plot for temperature")
plt.show()

med2 = df["rain"].median()
for i in df["rain"]:
    if(i<(q1 - 1.5*IqR) or i>(q3 + 1.5*IqR)):
        df["rain"] = df["rain"].replace(i,med2,regex = True)
plt.boxplot(df["rain"])
plt.title("Box plot for rain")
plt.show()