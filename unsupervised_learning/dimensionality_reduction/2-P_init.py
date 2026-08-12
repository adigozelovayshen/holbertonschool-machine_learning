#!/usr/bin/env python3
"""
Initialize t-SNE module
"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Parameters:
    - X (numpy.ndarray): Dataset of shape (n, d)
    - perplexity (float): Perplexity for Gaussian distributions

    Returns:
    - D (numpy.ndarray): Squared pairwise distance matrix of shape (n, n)
    - P (numpy.ndarray): P affinities matrix initialized to zeros of shape (n, n)
    - betas (numpy.ndarray): Beta values initialized to ones of shape (n, 1)
    - H (float): Shannon entropy for the perplexity with base 2
    """
    n, d = X.shape

    # Kvadratik Evklid məsafəsi: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*(x_i . x_j)
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X.reshape(-1, 1) + sum_X.reshape(1, -1) - 2 * np.matmul(X, X.T)
    np.fill_diagonal(D, 0)

    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
