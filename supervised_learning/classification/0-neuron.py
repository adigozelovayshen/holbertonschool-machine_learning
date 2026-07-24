#!/usr/bin/env python3
"""
Single Neuron Module
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initializer for the Neuron class

        Parameters:
        - nx: number of input features to the neuron

        Raises:
        - TypeError: if nx is not an integer
        - ValueError: if nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
