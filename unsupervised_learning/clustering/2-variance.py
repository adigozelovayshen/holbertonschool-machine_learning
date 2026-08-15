#!/usr/bin/env python3
"""Calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set.
    - X: numpy.ndarray of shape (n, d) containing the data set
    - C: numpy.ndarray of shape (k, d) containing centroid means
    Returns: var (total variance), or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate distances from each data point to each centroid: (n, k)
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)

    # Find the minimum distance squared for each data point to its closest centroid
    min_distances = np.min(distances, axis=1)

    # Total variance is the sum of squared distances
    return np.sum(min_distances)
