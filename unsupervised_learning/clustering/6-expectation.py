#!/usr/bin/env python3
"""Module that calculates the E-step in EM algorithm."""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM.
    - X: numpy.ndarray of shape (n, d)
    - pi: numpy.ndarray of shape (k,)
    - m: numpy.ndarray of shape (k, d)
    - S: numpy.ndarray of shape (k, d, d)
    Returns: g, l, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    k, d = m.shape
    n = X.shape[0]

    if pi.shape[0] != k or S.shape[0] != k or S.shape[1] != d or \
       S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    # Calculate weighted likelihoods for each cluster
    g = np.zeros((k, n))
    for i in range(k):
        # Calculate PDF for current cluster
        p = pdf(X, m[i], S[i])
        if p is None:
            return None, None
        g[i] = pi[i] * p

    # Total likelihood: sum over clusters for each point
    total_likelihood = np.sum(g, axis=0)
    
    # Calculate Log Likelihood
    l = np.sum(np.log(total_likelihood))
    
    # Posterior probabilities: g_ik = (pi_i * P(x_j|theta_i)) / sum_l(pi_l * P(x_j|theta_l))
    g /= total_likelihood

    return g, l
