#!/usr/bin/env python3
"""Performs element-wise operations using numpy"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    Returns a tuple containing the results in that order
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
