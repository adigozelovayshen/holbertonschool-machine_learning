#!/usr/bin/env python3
"""
Neuron Module with Gradient Descent
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

    def forward_prop(self, X):
        """Calculates forward prop"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates cost"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent
        """
        m = Y.shape[1]
        dz = A - Y
        dw = 1 / m * np.matmul(dz, X.T)
        db = 1 / m * np.sum(dz)
        self.__W -= alpha * dw
        self.__b -= alpha * db
