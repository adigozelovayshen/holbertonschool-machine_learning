#!/usr/bin/env python3
"""Calculates positional encoding for a transformer model."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len: integer representing the maximum sequence length
        dm: integer representing the model depth

    Returns:
        numpy.ndarray of shape (max_seq_len, dm) containing positional
        encoding vectors.
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]

    # Angle rates calculation: pos / (10000 ^ (2i / dm))
    angle_rates = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))

    pe = np.zeros((max_seq_len, dm))

    # Apply sin to even indices (2i) and cos to odd indices (2i+1)
    pe[:, 0::2] = np.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return pe
