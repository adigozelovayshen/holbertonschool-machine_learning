#!/usr/bin/env python3
"""Normal paylanması modulu"""


class Normal:
    """Normal paylanmasını təmsil edən sinif"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Sinif konstruktoru"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Mean hesabı: sum(x) / n
            self.mean = float(sum(data) / len(data))

            # Stddev hesabı: sqrt(sum((x - mean)**2) / n)
            sum_diff_sq = sum([(x - self.mean) ** 2 for x in data])
            self.stddev = float((sum_diff_sq / len(data)) ** 0.5)
