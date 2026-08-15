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

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for _ in range(iterations):
        # Calculate distances using broadcasting (ensures stable precision)
        distances = np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2))
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        # Update centroids
        new_C = []
        for j in range(k):
            members = X[clss == j]
            if len(members) == 0:
                new_C.append(np.random.uniform(low, high, size=(d,)))
            else:
                new_C.append(np.mean(members, axis=0))
        C = np.array(new_C)

        if np.all(C == C_prev):
            break

    return C, clss
