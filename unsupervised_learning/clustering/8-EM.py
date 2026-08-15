#!/usr/bin/env python3
"""Module that performs Expectation-Maximization for GMM."""
import numpy as np
initialize = __import__('6-initialize').initialize
expectation = __import__('7-expectation').expectation
maximization = __import__('8-maximization').expectation


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Performs expectation maximization for a GMM."""
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
    l_prev = 0

    for i in range(iterations + 1):
        g, ll = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {ll}")
        
        if abs(ll - l_prev) <= tol and i > 0:
            break
        
        l_prev = ll
        if i < iterations:
            pi, m, S = maximization(X, g)

    if verbose:
        print(f"Log Likelihood after {i} iterations: {ll}")

    return pi, m, S, g, ll
