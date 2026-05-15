#!/usr/bin/env python3
"""Eksponensial paylanması modulu"""


class Exponential:
    """Eksponensial paylanmasını təmsil edən sinif"""

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
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """Verilmiş x zaman kəsiyi üçün PDF dəyərini hesablayır"""
        if x < 0:
            return 0
        e = 2.7182818285
        pdf_val = self.lambtha * (e ** (-self.lambtha * x))
        return pdf_val

    def cdf(self, x):
        """Verilmiş x zaman kəsiyi üçün CDF dəyərini hesablayır"""
        if x < 0:
            return 0
        e = 2.7182818285
        # F(x) = 1 - e^(-lambtha * x)
        cdf_val = 1 - (e ** (-self.lambtha * x))
        return cdf_val
