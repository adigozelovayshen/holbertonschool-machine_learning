#!/usr/bin/env python3
"""Module that performs the expectation maximization for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    prev_l = 0
    g, l = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(l))

    for i in range(1, iterations + 1):
        prev_l = l
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, l = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or abs(l - prev_l) <= tol):
            print("Log Likelihood after {} iterations: {}".format(i, l))

        if abs(l - prev_l) <= tol:
            if verbose and i % 10 != 0:
                print("Log Likelihood after {} iterations: {}".format(i, l))
            break

    return pi, m, S, g, l
