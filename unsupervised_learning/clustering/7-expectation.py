#!/usr/bin/env python3
"""Module that calculates the expectation step in the EM algorithm for a GMM."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d) or not np.isclose(np.sum(pi), 1):
        return None, None

    P = np.zeros((k, n))
    for i in range(k):
        P[i] = pdf(X, m[i], S[i])
        if P[i] is None:
            return None, None
        P[i] *= pi[i]

    sum_P = np.sum(P, axis=0)
    if np.any(sum_P == 0):
        return None, None

    g = P / sum_P
    log_likelihood = np.sum(np.log(sum_P))

    return g, log_likelihood
