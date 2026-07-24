#!/usr/bin/env python3
"""
Pooling Forward Propagation Module
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network

    Parameters:
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - kernel_shape: tuple of (kh, kw)
    - stride: tuple of (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            slice_A = A_prev[:, v_start:v_end, h_start:h_end, :]

            if mode == 'max':
                A[:, i, j, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, i, j, :] = np.mean(slice_A, axis=(1, 2))

    return A
