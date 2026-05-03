#!/usr/bin/env python3
"""
Concatenates two matrices along a specific axis
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Returns a new matrix or None if shapes are incompatible
    """
    # Əgər axis 0-dırsa, birbaşa birinci səviyyədə birləşdiririk
    if axis == 0:
        # Ölçü uyğunluğunu yoxlamaq üçün (axis=0 istisna olmaqla digərləri)
        # Matrislərin içindəki elementlərin tipini və ölçüsünü rekursiv yoxlamalıyıq
        # Sadəlik üçün əvvəlcə formaların (shape) eyni olduğunu yoxlayaq
        if not can_concatenate(mat1, mat2, axis):
            return None
        return mat1 + mat2

    # Əgər axis 0-dan böyükdürsə, daha dərinə (rekursiya) gedirik
    if not isinstance(mat1, list) or not isinstance(mat2, list) or \
       len(mat1) != len(mat2):
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
        # Axis 0 üçün mat1 və mat2-nin daxili strukturu (shape) eyni olmalıdır
        # Əgər onlar list deyillərsə (skalyar), deməli eynidirlər
        if not isinstance(mat1, list) and not isinstance(mat2, list):
            return True
        # Biri list digəri yoxdursa, uyğun deyil
        if type(mat1) is not type(mat2):
            return False
        # Birinci səviyyədən sonrakı bütün ölçülər (shape) eyni olmalıdır
        return get_shape(mat1)[1:] == get_shape(mat2)[1:]

    # Əgər daha dərin oxları yoxlayırıqsa
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
