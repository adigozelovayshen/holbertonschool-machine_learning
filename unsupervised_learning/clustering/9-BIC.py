#!/usr/bin/env python3
"""Module that finds the best number of clusters for a GMM using BIC."""
import numpy as np


def kmeans(X, k):
    """K-means clustering."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    n, d = X.shape
    idx = np.random.choice(n, k, replace=False)
    C = X[idx].copy()
    while True:
        D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        labels = np.argmin(D, axis=1)
        C_new = np.array([X[labels == i].mean(axis=0) if np.any(labels == i)
                          else X[np.random.choice(n)] for i in range(k)])
        if np.all(C == C_new):
            break
        C = C_new
    return C, labels


def pdf(X, mean, S):
    """Calculates the probability density function of a Gaussian distribution."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(mean, np.ndarray) or len(mean.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    n, d = X.shape
    if mean.shape[0] != d or S.shape != (d, d):
        return None
    det = np.linalg.det(S)
    if det == 0:
        return None
    inv = np.linalg.inv(S)
    norm = 1.0 / (np.power((2 * np.pi), d / 2) * np.sqrt(det))
    diff = X - mean
    exponent = -0.5 * np.sum(np.dot(diff, inv) * diff, axis=1)
    return norm * np.exp(exponent)


def initialize(X, k):
    """Initializes Gaussian Mixture Model variables."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None
    n, d = X.shape
    C, _ = kmeans(X, k)
    pi = np.ones(k) / k
    S = np.tile(np.identity(d), (k, 1, 1))
    return pi, m := C, S


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM."""
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
        res = pdf(X, m[i], S[i])
        if res is None:
            return None, None
        P[i] = res * pi[i]
    sum_P = np.sum(P, axis=0)
    if np.any(sum_P == 0):
        return None, None
    g = P / sum_P
    log_likelihood = np.sum(np.log(sum_P))
    return g, log_likelihood


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k, n_g = g.shape
    if n != n_g or not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None
    pi = np.sum(g, axis=1) / n
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        sum_g = np.sum(g[i])
        if sum_g == 0:
            return None, None, None
        m[i] = np.dot(g[i], X) / sum_g
        diff = X - m[i]
        S[i] = np.dot(g[i] * diff.T, diff) / sum_g
    return pi, m, S


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
    if pi is None:
        return None, None, None, None, None
    l_prev = 0
    for i in range(iterations + 1):
        g, ll = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None
        if verbose and i % 10 == 0:
            print(f"Log Likelihood after {i} iterations: {ll}")
        if abs(ll - l_prev) <= tol and i > 0:
            break
        l_prev = ll
        if i < iterations:
            pi, m, S = maximization(X, g)
            if pi is None:
                return None, None, None, None, None
    if verbose:
        print(f"Log Likelihood after {i} iterations: {ll}")
    return pi, m, S, g, ll


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
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
    if kmin >= kmax or kmin > n or kmax > n:
        return None, None, None, None
    results = []
    log_likelihoods = []
    bics = []
    ks = range(kmin, kmax + 1)
    for k in ks:
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None:
            return None, None, None, None
        p = (k - 1) + (k * d) + int(k * d * (d + 1) / 2)
        bic = p * np.log(n) - 2 * ll
        results.append((pi, m, S))
        log_likelihoods.append(ll)
        bics.append(bic)
    best_idx = np.argmin(bics)
    return ks[best_idx], results[best_idx], np.array(log_likelihoods), \
        np.array(bics)
