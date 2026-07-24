#!/usr/bin/env python3
"""
Neuron Module with private attributes
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initializer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for __W"""
        return self.__W

    @property
    def b(self):
        """Getter for __b"""
        return self.__b

    @property
    def A(self):
        """Getter for __A"""
        return self.__A
