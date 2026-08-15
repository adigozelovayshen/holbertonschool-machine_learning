#!/usr/bin/env python3
"""Module that performs K-means clustering using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.
    
    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: the number of clusters
    
    Returns:
    - C: numpy.ndarray of shape (k, d) containing the centroid means
    - clss: numpy.ndarray of shape (n,) containing the cluster index for each point
    """
    # Initialize KMeans from sklearn.cluster
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(X)
    
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    
    return C, clss
