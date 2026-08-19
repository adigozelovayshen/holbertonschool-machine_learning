#!/usr/bin/env python3
"""
Module containing the GRUCell class
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        Parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        # Wr MUST be initialized BEFORE Wz to match random seed sequence
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.br = np.zeros((1, h))
        self.bz = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Parameters:
            h_prev: numpy.ndarray of shape (m, h) with previous hidden state
            x_t: numpy.ndarray of shape (m, i) with data input for the cell

        Returns:
            h_next: next hidden state
            y: output of the cell
        """
        # Concatenate x_t and h_prev: shape (m, i + h)
        concat_x_h = np.concatenate((x_t, h_prev), axis=1)

        # Reset gate
        r = 1 / (1 + np.exp(-(np.matmul(concat_x_h, self.Wr) + self.br)))

        # Update gate
        z = 1 / (1 + np.exp(-(np.matmul(concat_x_h, self.Wz) + self.bz)))

        # Candidate hidden state
        concat_reset = np.concatenate((x_t, r * h_prev), axis=1)
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output calculation with Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
