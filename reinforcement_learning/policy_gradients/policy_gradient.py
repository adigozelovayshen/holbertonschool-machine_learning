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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on state and weight matrix.

    Parameters:
    - state: matrix representing the current observation of the environment
    - weight: matrix of random weights

    Returns:
    - action: selected action index
    - grad: Monte-Carlo policy gradient
    """
    if state.ndim == 1:
        state = state[np.newaxis, :]

    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    grad = np.dot(state.T, dsoftmax)

    return action, grad
