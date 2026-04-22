#!/usr/bin/env python3
"""Function to calculate the integral of a polynomial"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None

    for x in poly:
        if not isinstance(x, (int, float)):
            return None

    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i + 1)
        if val % 1 == 0:
            val = int(val)
        integral.append(val)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
