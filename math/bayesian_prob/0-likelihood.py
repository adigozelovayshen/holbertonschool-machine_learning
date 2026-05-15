#!/usr/bin/env python3
"""Bayesian ehtimalı - Likelihood hesablama modulu"""
import numpy as np
import math


def likelihood(x, n, P):
    """
    Binomial paylanma üçün verilən x və n əsasında P ehtimallarının
    likelihood dəyərlərini hesablayır.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Kombinasiya (nCk) = n! / (x! * (n-x)!)
    # Standart math modulundan istifadə edirik
    fact_n = math.factorial(n)
    fact_x = math.factorial(x)
    fact_nx = math.factorial(n - x)
    combination = fact_n / (fact_x * fact_nx)

    # Likelihood = combination * (P^x) * ((1-P)^(n-x))
    lh = combination * (P ** x) * ((1 - P) ** (n - x))

    return lh
