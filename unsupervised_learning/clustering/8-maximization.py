#!/usr/bin/env python3
"""Module that calculates the maximization step in the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape
    if n != n_g:
        return None, None, None
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    pi = np.sum(g, axis=1) / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        sum_g = np.sum(g[i])
        if sum_g == 0:
            return None, None, None
        m[i] = np.dot(g[i], X) / sum_g
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / sum_g

    return pi, m, S
