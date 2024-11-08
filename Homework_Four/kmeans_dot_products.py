'''
Implement a modified k-means algorithm that takes the dot product matrix 𝑲 rather than
the data matrix 𝑿 as input. Your implementation should be in a function with the following
header:
def fit_kmeans_dot_products(K, n_clusters, max_iter=300)
It should output the cluster labels in the same form as the scikit-learn implementation in
sklearn.cluster.KMeans. Submit your code as kmeans_dot_products.py. Hints
'''


import numpy as np 

def fit_kmeans_dot_products(K, n_clusters, max_iter=300):
     num_samples = K.shape[0] #find the number of samples from the first part of the K matrix 
     cluster = np.random.randint(0, n_clusters, num_samples) #initialize value for the cluster we need to return 
     

     for i in range(max_iter):
        distance_matrix = np.zeros((num_samples, n_clusters))

        for j in range(n_clusters): 
            index = np.where(labels == j)[0] #get indices of samples in cluster 


            distance_matrix[i, j] = K[i, i] - (2/j) * np.sum(K[:,j] * K[:,j]) + (1/(j**2))*np.sum(K[j,:][:,j])
        real_cluster = np.argmin(distance_matrix)

     return real_cluster

