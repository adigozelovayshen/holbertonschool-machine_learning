#!/usr/bin/env python3
"""Poisson paylanması modulu"""


class Poisson:
    """Poisson paylanmasını təmsil edən sinif"""

    def __init__(self, data=None, lambtha=1.):
        """Sinif konstruktoru"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Verilmiş k uğur sayı üçün PMF dəyərini hesablayır"""
        if k < 0:
            return 0
        
        k = int(k)
        e = 2.7182818285
        
        # Faktorial hesablama
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
            
        # PMF düsturu: (e^-lambda * lambda^k) / k!
        pmf_val = (e ** -self.lambtha * self.lambtha ** k) / factorial
        return pmf_val
