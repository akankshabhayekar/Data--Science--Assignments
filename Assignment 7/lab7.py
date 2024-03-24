# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 19:36:19 2022

@author: akank
"""
#NAME=AKANKSHA BHAYEKAR
#ROLL.NO=B21148
#MOBILE NO=8177936098

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# reading the desired csv file
iris = pd.read_csv(r"C:\Users\akank\Documents\ic160\Iris.csv")
test = iris['Species']
org = []
for i in test:
    if i == 'Iris-setosa':
        org.append(0)
    elif i == 'Iris-virginica':
        org.append(2)
    else:
        org.append(1)

# dropping the 5th attribute
iris = iris.drop(['Species'], axis=1)
names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# 1
# reducing the data from 4 dimensional to 2 dimensional
pca = PCA(n_components=2)
reduced = pd.DataFrame(pca.fit_transform(iris), columns=['x1', 'x2'])
val, vec = np.linalg.eig(iris.corr().to_numpy())
c = np.linspace(1, 4, 4)
plt.plot(c, [round(i,3) for i in val], color='blue')
plt.xticks(np.arange(min(c), max(c)+1, 1.0))
plt.xlabel('Components')
plt.ylabel('Eigen Values')
plt.title('Eigen Values vs Components')
plt.show()

print("2:-")
# 2
K = 3  # given
kmeans = KMeans(n_clusters=K)
kmeans.fit(reduced)
k_pred = kmeans.predict(reduced)
reduced['k_cluster'] = kmeans.labels_
kcentres = kmeans.cluster_centers_
# 2-a.
# plotting the scatter plot
plt.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=k_pred, cmap='rainbow', s=15)
plt.scatter([kcentres[i][0] for i in range(K)], [kcentres[i][1] for i in range(K)], c='black', marker='o',
            label='cluster centres')
plt.legend()
plt.title('Data Points (K-Means)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
print("2-a.")
print("The Scatter Plot has been shown above")

# 2-b.
print('2-b.')
print('The Distortion Measure for k =3 is', round(kmeans.inertia_, 3))


# 2-c.
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)  # print(contingency_matrix)
    # print(contingency_matrix)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


print('2-c.')
print('The Purity Score for k =3 is', round(purity_score(org, k_pred), 3))

# 3
reduced = reduced.drop(['k_cluster'], axis=1)
Ks = [2, 3, 4, 5, 6, 7]
kdistortion = []
kpurity = []
for k in Ks:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reduced)
    kdistortion.append(round(kmeans.inertia_, 3))
    kpurity.append(round(purity_score(org, kmeans.predict(reduced)), 3))

print("\n")
print('3:-')
print('The Distortion Measures are', kdistortion)
print('The Purity Scores are', kpurity)

# plotting K vs distortion measure
plt.plot(Ks, kdistortion, color='violet')
plt.title('Distortion Measure vs K')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()
print("\n")
print("4:-")
# 4
# building gmm
gmm = GaussianMixture(n_components=K, random_state=42).fit(reduced)
gmm_pred = gmm.predict(reduced)
reduced['gmm_cluster'] = gmm_pred
gmmcentres = gmm.means_
# 4-a.
# plotting the scatter plot
plt.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=gmm_pred, cmap='rainbow', s=15)
plt.scatter([gmmcentres[i][0] for i in range(K)], [gmmcentres[i][1] for i in range(K)], c='black', marker='o',
            label='cluster centres')
plt.legend()
plt.title('Data Points(GMM)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
print("4-a.")
print("The Scatter Plot has been shown above")
# 4-b.
reduced = reduced.drop(['gmm_cluster'], axis=1)
print('4-b.')
print('The Distortion Measure for k =3 is', round(gmm.score(reduced) * len(reduced), 3))

# 4-c.
print('4-c.')
print('The Purity Score for k =3 is', round(purity_score(org, gmm_pred), 3))

print("\n")
print("5:-")
# 5
total_log = []
gmpurity = []
for k in Ks:
    gmm = GaussianMixture(n_components=k, random_state=42).fit(reduced)
    total_log.append(round((gmm.score(reduced) * len(reduced)), 3))
    gmpurity.append(round(purity_score(org, gmm.predict(reduced)), 3))


print('The Distortion Measures are', total_log)
print('The Purity Scores are', gmpurity)

# plotting K vs distortion measure
plt.plot(Ks, total_log, color='red')
plt.title('Distortion Measure vs K')
plt.xlabel('K')
plt.ylabel('Distortion Measure')
plt.show()
print("\n")
print("6:-")
# 6

eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(reduced)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for eps={eps[i]} and min_samples={min_samples[i]} is',
          round(purity_score(org, DBSCAN_predictions), 3))
    plt.scatter(reduced[reduced.columns[0]], reduced[reduced.columns[1]], c=DBSCAN_predictions, cmap='flag', s=15)
    plt.title(f'Data Points for eps={eps[i]} and min_samples={min_samples[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

