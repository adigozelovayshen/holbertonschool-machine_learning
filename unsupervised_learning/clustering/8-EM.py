#!/usr/bin/env python3
"""
Expectation Maximization for GMM
"""
import numpy as np
initialize = __import__('2-variance').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Takes a dataset and computes Expectation Maximization for a GMM

    Parameters:
    - X: numpy.ndarray of shape (n, d) dataset
    - k: positive integer for number of clusters
    - iterations: maximum number of iterations
    - tol: tolerance for EM algorithm
    - verbose: boolean to print info

    Returns:
    - pi, m, S, g, log_l or None, None, None, None, None on failure
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

    l_old = 0
    g, log_l = expectation(X, pi, m, S)
    if g is None or log_l is None:
        return None, None, None, None, None

    for i in range(iterations):
        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {}".format(
                i, log_l.round(5)))

        l_old = log_l

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        g, log_l = expectation(X, pi, m, S)
        if g is None or log_l is None:
            return None, None, None, None, None

        if abs(log_l - l_old) <= tol:
            break

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1 if abs(log_l - l_old) <= tol else iterations,
            log_l.round(5)))

    return pi, m, S, g, log_l
