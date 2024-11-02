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
Hint: The sklearn.cluster.SpectralClustering object provides a good implementation
for normalized cut spectral clustering. The sklearn.cluster.AgglomerativeClustering
provides an implementation for agglomerative hierarchical clustering but may not be as fully fea-
tured as the implementation in scipy.cluster.hierarchy, which includes a function to plot
the dendrogram.
'''