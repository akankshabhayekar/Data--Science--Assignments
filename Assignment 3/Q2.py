# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 22:00:32 2022

@author: akank
"""

import numpy as np
import matplotlib.pyplot as plt

# Creating a Covarience Matrix

cm = np.array([[13,-3],[-3,5]])

# Finding the Eigenvalues and the Eigenvectors

w ,v = np.linalg.eigh(cm)

# Individual Eigenvectors

v1 = v[:,0]
v2 = v[:,1]

np.random.seed(10)

# gGenerating a Bi-Variate Synthetic Gaussian Data

x = np.random.multivariate_normal([0,0],cm,size=1000)

# PART 1

# Plotting the Scatterplot for the Synthetic Data

fig, ax = plt.subplots(figsize =(14, 9))
plt.scatter(x[:,0],x[:,1],marker='x',color='blue')
plt.title('Scatterplot of the Bi-Variate Synthetic Gaussian Data',size=22)
ax.grid(b = True, color ='grey',linestyle ='-.', linewidth = 0.7,alpha = 1)
plt.show()

# PART 2

# Plotting the EigenDirections onto the Scatterplot
plt.scatter(x[:,0],x[:,1],marker='+',color='blue')
plt.title('Scatterplot along with the Eigendirections',size=22)
plt.arrow(0,0,-10*v[0,0],-10*v[1,0],head_width = 0.4,width = 0.1,color='red')
plt.arrow(0,0,10*v[0,1],10*v[1,1],head_width = 0.4,width = 0.1,color='red')

plt.show()

# PART 3 (a)

# Projecting the data onto the EigEnvectors 1

d1 = np.dot(x,v1)
d1 = np.dot(d1.reshape(1000,1),v1.reshape(1,2))
plt.scatter(x[:,0],x[:,1],marker='+',color='black')
plt.scatter(d1[:,0],d1[:,1],marker='o',color='green',s=100)
plt.title('Projecting the Data along the Vector 1',size=22)
plt.arrow(0,0,-10*v[0,0],-10*v[1,0],head_width = 0.4,width = 0.1,color='red')
plt.arrow(0,0,10*v[0,1],10*v[1,1],head_width = 0.4,width = 0.1,color='red')
plt.show()

# PART 3 (b)

# Projecting the data onto Eigenvectors 2

d2 = np.dot(x,v2)
d2 = np.dot(d2.reshape(1000,1),v2.reshape(1,2))
plt.scatter(x[:,0],x[:,1],marker='+',color='blue')
plt.scatter(d2[:,0],d2[:,1],marker='o',color='black',s=100)
plt.title('Projecting the Data along the Vector 2',size=22)