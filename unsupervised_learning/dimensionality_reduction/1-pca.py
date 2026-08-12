#!/usr/bin/env python3
"""
PCA v2 module
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Parameters:
    - X (numpy.ndarray): Data matrix of shape (n, d)
    - ndim (int): New dimensionality of transformed X

    Returns:
    - T (numpy.ndarray): Transformed version of X with shape (n, ndim)
    """
    # 1. Məlumatı sıfır mərkəzli edin (Zero-center the data)
    X_m = X - np.mean(X, axis=0)

    # 2. SVD tətbiq edin
    u, s, vh = np.linalg.svd(X_m)

    # 3. İlk 'ndim' qədər əsas komponenti (çəki matrisini) götürün
    W = vh[:ndim].T

    # 4. Məlumatı yeni ölçüyə köçürün
    T = np.matmul(X_m, W)

    return T
