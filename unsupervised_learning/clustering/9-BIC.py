#!/usr/bin/env python3
"""
Bayesian Information Criterion (BIC) for GMM
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using BIC

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - kmin: minimum number of clusters to check for (inclusive)
    - kmax: maximum number of clusters to check for (inclusive)
    - iterations: maximum number of iterations for EM algorithm
    - tol: tolerance for EM algorithm
    - verbose: boolean to print EM information

    Returns:
    - best_k, best_result, l, b, or None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < 1 or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    k_values = kmax - kmin + 1
    l = np.zeros(k_values)
    b = np.zeros(k_values)
    results = []

    for idx, k in enumerate(range(kmin, kmax + 1)):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or log_l is None:
            return None, None, None, None

        l[idx] = log_l
        results.append((pi, m, S))

        # p = number of parameters
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        # BIC = p * ln(n) - 2 * l
        b[idx] = p * np.log(n) - 2 * log_l

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
