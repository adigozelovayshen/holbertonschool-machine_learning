#!/usr/bin/env python3
"""Module that finds the best number of clusters for a GMM using BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    if kmin > kmax or kmin > n or kmax > n:
        return None, None, None, None

    results = []
    l_list = []
    b_list = []
    ks = range(kmin, kmax + 1)

    for k in ks:
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None

        p = (k - 1) + (k * d) + int(k * d * (d + 1) / 2)
        bic = p * np.log(n) - 2 * log_l

        l_list.append(log_l)
        b_list.append(bic)
        results.append((pi, m, S))

    l = np.array(l_list)
    b = np.array(b_list)

    best_index = np.argmin(b)
    best_k = ks[best_index]
    best_result = results[best_index]

    return best_k, best_result, l, b
