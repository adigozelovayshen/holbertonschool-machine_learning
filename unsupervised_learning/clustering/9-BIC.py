#!/usr/bin/env python3
"""
Bayesian Information Criterion (BIC) module
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: positive integer containing minimum number of clusters
    - kmax: positive integer containing maximum number of clusters
    - iterations: positive integer containing maximum iterations
    - tol: non-negative float containing tolerance
    - verbose: boolean that determines if EM should print info

    Returns:
    - best_k, best_result, l, b, or None, None, None, None on failure
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if type(kmax) is not int or kmax <= 0 or kmin > kmax:
        return None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or log_l is None:
            return None, None, None, None

        idx = k - kmin
        l[idx] = log_l
        results.append((pi, m, S))

        # p = number of parameters
        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        # BIC = p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
