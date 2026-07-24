#!/usr/bin/env python3
"""
Pooling Back Propagation Module
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network

    Parameters:
    - dA: numpy.ndarray of shape (m, h_new, w_new, c_new)
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c)
    - kernel_shape: tuple of (kh, kw)
    - stride: tuple of (sh, sw)
    - mode: 'max' or 'avg'

    Returns:
    - dA_prev: partial derivatives with respect to previous layer
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            if mode == 'max':
                for n in range(m):
                    for k in range(c_new):
                        slice_A = A_prev[n, v_start:v_end, h_start:h_end, k]
                        mask = (slice_A == np.max(slice_A))
                        dA_prev[n, v_start:v_end, h_start:h_end, k] += (
                            mask * dA[n, i, j, k]
                        )
            elif mode == 'avg':
                avg_dA = dA[:, i:i+1, j:j+1, :] / (kh * kw)
                dA_prev[:, v_start:v_end, h_start:h_end, :] += avg_dA

    return dA_prev
