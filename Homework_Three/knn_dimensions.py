'''
Some machine learning methods, such as nearest neighbors-based methods, are susceptible to the
curse of dimensionality. This means that their accuracy degrades significantly as the number of
dimensions (features) increases, as shown in the figure below.
To verify that this is the case for nearest neighbors, complete the following. Generate 1,000 ran-
dom data points in 𝑑 dimensions, where each dimension follows a uniform random variable in
[−1, 1]. Plot the following measures as you increase the number of dimensions 𝑑 from 2 to 10:
a) The fraction of data points within the unit hypersphere, i.e. the fraction of data points with
distance ≤ 1 from the origin (all zero coordinates). This measures roughly what fraction
of data points are close to a typical (all zero) data point.
b) The mean distance between a data point and its 1-nearest neighbor divided by the mean
distance between any pair of data points. This measures how close a nearest neighbor is
relative to a randomly selected data point. (As the number of dimensions increases, the
mean distance between any pair of data points also increases, so we divide by this to pro-
vide a fair comparison.)
Describe how the trend in these measures compares to what you would expect for a method
susceptible to the curse of dimensionality. Submit your code as knn_dimensions.py.
Hint: Use the sklearn.neighbors.NearestNeighbors class to compute the 1-nearest neigh-
bor distances. Use the scipy.spatial.distance.pdist() function to compute the pairwise
distances between all data points.
'''

import numpy as np 
import matplotlib.pyplot as plt

#function for calculating data points within the unit hypersphere 
def hypersphere(data):

    distances_from_origin = np.linalg.norm(data, axis=1)
    
    # Count how many points are within the unit hypersphere (distance <= 1)
    fraction = np.sum(distances_from_origin <= 1) / len(data)
    
    return fraction


#generating data points in d dimensions 
d = [2, 3, 4, 5, 6, 7, 8, 9, 10]
fraction = []

for i in d: 
    data = np.random.uniform(low=-1, high=1, size=(1000, i))

    fraction.append(hypersphere(data))
    i = i + 1


plt.plot(d, fraction)
plt.title("Fraction of Datapoints within the Unit Hypersphere")
plt.xlabel("Dimensions")
plt.ylabel("Distance from Origin")
plt.show()

