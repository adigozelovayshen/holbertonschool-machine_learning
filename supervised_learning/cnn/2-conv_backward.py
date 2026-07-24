#!/usr/bin/env python3
"""
Convolutional Back Propagation Module
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network

    Parameters:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new)
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
    - padding: "same" or "valid"
    - stride: tuple of (sh, sw)

    Returns:
    - dA_prev, dW, db
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    padded_A = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    padded_dA = np.zeros_like(padded_A)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            for k in range(c_new):
                slice_A = padded_A[:, v_start:v_end, h_start:h_end, :]
                dZ_val = dZ[:, i:i+1, j:j+1, k:k+1]

                padded_dA[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, k] * dZ_val
                )
                dW[:, :, :, k] += np.sum(
                    slice_A * dZ_val,
                    axis=0
                )

    if padding == "same":
        dA_prev = padded_dA[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = padded_dA

    return dA_prev, dW, db
