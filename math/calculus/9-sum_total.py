#!/usr/bin/env python3
"""Sum of squares calculation without loops"""


def summation_i_squared(n):
    """Calculates the sum of i^2 from 1 to n using Faulhaber's formula"""
    if not isinstance(n, int) or n < 1:
        return None
    
    result = (n * (n + 1) * (2 * n + 1)) // 6
    return result
