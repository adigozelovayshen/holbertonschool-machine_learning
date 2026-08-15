#!/usr/bin/env python3
"""Module that initializes GMM variables."""
import numpy as np
kmeans = __import__('2-kmeans').kmeans


def initialize(X, k):
    """
    Initializes Gaussian Mixture Model variables.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape
    C, m = kmeans(X, k)
    pi = np.ones(k) / k
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
