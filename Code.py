import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random
import math
import numpy as np
import pandas as pd

N_CLUSTERS = 2
MIN_DIST = 50
BATCH_SIZE = 100
x = np.array(pd.read_csv('data.csv'))
DATA_SIZE = len(x)
MIN_CLUSTER = 1/5

def get_centers(arr, finarr, n_clusters):
    centers = []
    centers_val = [0]*n_clusters
    for i in range(n_clusters):
        centers.append([0, 0])
    for i in range(len(finarr)):
        a = centers[finarr[i]]
        centers_val[finarr[i]] += 1
        a[0] += arr[i][0]
        a[1] += arr[i][1]
    for i in range(n_clusters):
        centers[i][0] /= centers_val[i]
        centers[i][1] /= centers_val[i]
    return np.array(centers)
def is_enough(arr, finarr, n_clusters):
    val = [0] * n_clusters
    for i in finarr:
        val[i] += 1
    for i in range(n_clusters):
        for j in range(n_clusters):
            if val[i]/val[j] < MIN_CLUSTER:
                return False
    return True
def is_close(centers):
    for i in range(N_CLUSTERS):
        for j in range(i+1, N_CLUSTERS):
            length = math.pow(centers[i][0]-centers[j][0], 2) + math.pow(centers[i][1]-centers[j][1], 2)
            length = math.sqrt(length)
            if length < MIN_DIST:
                return False
    return True



labels = np.array([])
centers = np.array([])

x_batch = np.zeros((BATCH_SIZE, 2))
for i in range(BATCH_SIZE):
    x_batch[i] = x[random.randint(0, DATA_SIZE-1)]
batch_aggl = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="single").fit(x_batch)
batch_centers = get_centers(x_batch, batch_aggl.labels_, N_CLUSTERS)
centers = np.copy(batch_centers)

if not is_close(batch_centers) and is_enough(batch_centers, batch_aggl.labels_, N_CLUSTERS):
    kmeans = KMeans(n_clusters = N_CLUSTERS, init = centers, n_init=1).fit(x)
    labels = kmeans.labels_
else:
    aggl = AgglomerativeClustering(n_clusters =N_CLUSTERS, linkage="single").fit(x)
    labels = aggl.labels_
"""
colors = ['red', 'magenta', 'yellow', 'orange', 'blue', 'white', 'brown']
for i in range(DATA_SIZE):
    plt.scatter(x[i][0], x[i][1], c=colors[labels[i]])

for i in range(N_CLUSTERS):
    plt.scatter(centers[i][0], centers[i][1], c=colors[i], edgecolors='black')
plt.show()"""