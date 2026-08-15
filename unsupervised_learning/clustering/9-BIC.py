#!/usr/bin/env python3
"""Module that finds the best number of clusters for a GMM using BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
        return None, None, None, None
    
    n, d = X.shape
    if kmax is None:
        kmax = n
    if kmin > kmax or kmin > n:
        return None, None, None, None

    results, log_likelihoods, bics = [], [], []
    ks = range(kmin, kmax + 1)
    
    for k in ks:
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        
        if pi is None:
            # Uğursuz olduqda sonsuzluq əlavə edirik ki, 
            # np.argmin bunu seçməsin, amma proses qırılmasın
            log_likelihoods.append(-np.inf)
            bics.append(np.inf)
            results.append((None, None, None))
        else:
            p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
            bic = p * np.log(n) - 2 * ll
            log_likelihoods.append(ll)
            bics.append(bic)
            results.append((pi, m, S))

    log_likelihoods = np.array(log_likelihoods)
    bics = np.array(bics)
    
    # Əgər hamısı None olubsa (heç bir k işləməyibsə)
    if np.all(bics == np.inf):
        return None, None, None, None

    best_idx = np.argmin(bics)
    return ks[best_idx], results[best_idx], log_likelihoods, bics
