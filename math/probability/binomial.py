#!/usr/bin/env python3
"""Binomial paylanması modulu"""


class Binomial:
    """Binomial paylanmasını təmsil edən sinif"""

    def __init__(self, data=None, n=1, p=0.5):
        """Sinif konstruktoru"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Mean (mu) və Variance (sigma^2) hesabı
            mean = sum(data) / len(data)
            sum_diff_sq = sum([(x - mean) ** 2 for x in data])
            variance = sum_diff_sq / len(data)

            # p = 1 - (variance / mean)
            p_initial = 1 - (variance / mean)
            # n = round(mean / p)
            self.n = int(round(mean / p_initial))
            # Yeni p = mean / n
            self.p = float(mean / self.n)
