#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Returns a new matrix or None if shapes are incompatible
    """
    if axis == 0:
        if not can_concatenate(mat1, mat2, axis):
            return None
        return mat1 + mat2

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if len(mat1) != len(mat2):
        return None

    res = []
    for i in range(len(mat1)):
        concatenated = cat_matrices(mat1[i], mat2[i], axis - 1)
        if concatenated is None:
            return None
        res.append(concatenated)

    return res


def can_concatenate(mat1, mat2, axis):
    """
    Checks if two matrices can be concatenated along a specific axis
    """
    if axis == 0:
        if not isinstance(mat1, list) and not isinstance(mat2, list):
            return True
        if type(mat1) is not type(mat2):
            return False
        return get_shape(mat1)[1:] == get_shape(mat2)[1:]

    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return False
    if len(mat1) != len(mat2):
        return False

    return can_concatenate(mat1[0], mat2[0], axis - 1)


def get_shape(matrix):
    """
    Returns the shape of a matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
