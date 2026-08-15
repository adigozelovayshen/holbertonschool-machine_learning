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
        # 1. Assignment step
        dist = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(dist, axis=1)

        C_prev = C.copy()

        # 2. Update step
        for j in range(k):
            members = X[clss == j]
            if len(members) == 0:
                C[j] = np.random.uniform(low, high, size=(d,))
            else:
                C[j] = np.mean(members, axis=0)

        # Check convergence
        if np.all(C == C_prev):
            break

    return C, clss
