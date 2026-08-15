#!/usr/bin/env python3
"""Module that performs expectation maximization for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM.
    - X: numpy.ndarray of shape (n, d)
    - k: positive integer (number of clusters)
    - iterations: positive integer (max iterations)
    - tol: non-negative float (tolerance for early stopping)
    - verbose: boolean (whether to print log likelihood information)
    Returns: pi, m, S, g, l, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    g, prev_l = expectation(X, pi, m, S)
    if g is None or prev_l is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {:.5f}".format(prev_l))

    best_pi, best_m, best_S, best_g, best_l = pi, m, S, g, prev_l
    final_iter = 0

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        best_pi, best_m, best_S, best_g, best_l = pi, m, S, g, l
        final_iter = i

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if abs(l - prev_l) <= tol:
            break

        prev_l = l

    if verbose and final_iter % 10 != 0:
        print("Log Likelihood after {} iterations: {:.5f}".format(final_iter, best_l))

    return best_pi, best_m, best_S, best_g, best_l
