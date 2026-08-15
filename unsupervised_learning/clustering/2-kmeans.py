#!/usr/bin/env python3
"""Module that performs K-means clustering."""
import numpy as np


def kmeans(X, k):
    """
    Performs K-means clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    n, d = X.shape
    # Initialize centroids randomly from data points
    indices = np.random.choice(n, k, replace=False)
    C = X[indices].copy()

    while True:
        # Calculate distances from each point to each centroid
        D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        labels = np.argmin(D, axis=1)

        # Update centroids
        C_new = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else X[np.random.choice(n)]
            for i in range(k)
        ])

        if np.all(C == C_new):
            break
        C = C_new

    return C, labels
