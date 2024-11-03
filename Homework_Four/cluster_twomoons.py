'''
Load the two moons data provided in twomoons.csv. The data contains two interleaved half-
circle clusters (“moons”), where the third column denotes which cluster each example belongs to.
3
Using only the 2 features (first 2 columns), attempt to recover the true clusters using the following
algorithms:
a) K-means clustering.
b) Agglomerative hierarchical clustering.
c) Spectral clustering.
Specify the maximum adjusted Rand index you are able to achieve for each algorithm along
with the hyperparameter settings you used. Submit your code as cluster_twomoons.py.

Answer: 

1. K-means clustering

The maximum rand index I was able to achieve here was 0.30659591836734695. 
The hyperparameters I used were n_clusters=2, and n_init=10

2. Agglomerative heirarchical clustering

The maximum rand index I was able to achieve here was 0.5734638922888617. 
The hyperparameters I used were n_clusters=2, metric="euclidean", and linkage="average"

3. Spectral clustering 

The maximum rand index I was able to achieve here was 0.6364215062187837. 
The hyperparameters I used were n_clusters=2, affinity='rbf', random_state=1, n_init=10, gamma=10, n_neighbors=15, degree=3. 

'''
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering  
from sklearn.metrics import adjusted_rand_score 
import matplotlib.pyplot as plt 

moondata = np.genfromtxt('Homework_Four/twomoons.csv', delimiter=',', skip_header=1)
print(np.shape(moondata))
#separate features and targets
features = moondata[:, :2]
targets = moondata[:, 2]

'''for the KMeans
model = KMeans(n_clusters=2, random_state=1, n_init=10).fit(features)
prediction = model.predict(features).astype(int)
'''
'''for agglomerative clustering
model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='average')
prediction = model.fit_predict(features).astype(int)
'''

'''for spectral clustering'''
model = SpectralClustering(n_clusters=2, affinity='rbf', random_state=1, n_init=10, gamma=10, n_neighbors=15, degree=3)
prediction = model.fit_predict(features).astype(int)

print(adjusted_rand_score(targets, prediction))

