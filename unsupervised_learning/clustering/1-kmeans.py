#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    centroids = np.random.uniform(low, high, size=(k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset.
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing max number of iterations
    Returns: C, clss, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        # Calculate distances between all points and centroids
        # X: (n, 1, d), C: (1, k, d) -> distance: (n, k)
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2))
        
        # Assign each point to the closest centroid
        clss = np.argmin(distances, axis=1)

        # Store previous centroids to check for convergence
        C_prev = C.copy()

        # Update centroids
        C = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))
            for j in range(k)
        ])

        # Check if centroids haven't changed
        if np.all(C == C_prev):
            break

    return C, clss
