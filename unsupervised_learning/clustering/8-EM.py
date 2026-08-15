#!/usr/bin/env python3
"""
Module that performs the Expectation Maximization algorithm
for a Gaussian Mixture Model.
"""
import numpy as np


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """
    Performs the Expectation Maximization algorithm for a GMM.

    X is a numpy.ndarray of shape (n, d) containing the data points.
    k is the number of clusters.
    iterations is the maximum number of iterations.
    tol is the tolerance for convergence.
    verbose determines whether to print log likelihood information.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    if k > X.shape[0]:
        return None, None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None

    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape

    pi = np.full(k, 1 / k)

    m = np.random.uniform(
        low=np.min(X, axis=0),
        high=np.max(X, axis=0),
        size=(k, d)
    )

    S = np.tile(np.eye(d), (k, 1, 1))

    prev_log_l = 0
    log_l = 0

    for i in range(iterations):
        # Expectation step
        pdf = np.zeros((n, k))

        for j in range(k):
            try:
                det = np.linalg.det(S[j])
                inv = np.linalg.inv(S[j])
            except np.linalg.LinAlgError:
                return None, None, None, None, None

            if det <= 0:
                return None, None, None, None, None

            diff = X - m[j]
            exponent = -0.5 * np.sum(
                diff @ inv * diff,
                axis=1
            )

            pdf[:, j] = (
                pi[j]
                * np.exp(exponent)
                / np.sqrt((2 * np.pi) ** d * det)
            )

        g = pdf / np.sum(pdf, axis=1, keepdims=True)

        # Maximization step
        Nk = np.sum(g, axis=0)

        pi = Nk / n
        m = (g.T @ X) / Nk[:, np.newaxis]

        for j in range(k):
            diff = X - m[j]
            S[j] = (
                (g[:, j, np.newaxis] * diff).T @ diff
                / Nk[j]
            )

        # Log likelihood
        total = np.sum(pdf, axis=1)

        if np.any(total <= 0):
            return None, None, None, None, None

        log_l = np.sum(np.log(total))

        if verbose and (i == 0 or i % 10 == 0):
            print(
                "Log Likelihood after {} iterations: {}".format(
                    i, log_l
                )
            )

        # Convergence
        if i > 0 and abs(log_l - prev_log_l) <= tol:
            if verbose and i % 10 != 0:
                print(
                    "Log Likelihood after {} iterations: {}".format(
                        i, log_l
                    )
                )
            return pi, m, S, g, log_l

        prev_log_l = log_l

    return pi, m, S, g, log_l
