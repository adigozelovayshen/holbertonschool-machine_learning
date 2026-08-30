#!/usr/bin/env python3
"""
Policy Gradient module
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix using softmax.

    Parameters:
    - matrix: numpy.ndarray representing state
    - weight: numpy.ndarray representing weights

    Returns:
    - numpy.ndarray containing probabilities for each action
    """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
