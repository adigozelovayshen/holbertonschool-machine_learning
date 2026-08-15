#!/usr/bin/env python3
"""Module that performs expectation maximization for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


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
    g, prev_ll = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after 0 iterations: {}".format(prev_ll))

    best_pi, best_m, best_S, best_g, best_ll = pi, m, S, g, prev_ll
    final_iter = 0

    for i in range(1, iterations + 1):
        pi, m, S = maximization(X, g)
        g, ll = expectation(X, pi, m, S)
        best_pi, best_m, best_S, best_g, best_ll = pi, m, S, g, ll
        final_iter = i

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, ll))

        if abs(ll - prev_ll) <= tol:
            break
        prev_ll = ll

    if verbose and final_iter % 10 != 0:
        print("Log Likelihood after {} iterations: {}".format(
            final_iter, best_ll))

    return best_pi, best_m, best_S, best_g, best_ll
