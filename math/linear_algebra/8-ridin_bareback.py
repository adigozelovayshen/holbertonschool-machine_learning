#!/usr/bin/env python3
"""Performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Multiplies two matrices and returns a new matrix"""
    # Matris vurulması şərti: mat1 sütun sayı == mat2 sətir sayı
    if len(mat1[0]) != len(mat2):
        return None

    # Nəticə matrisinin ölçüləri: len(mat1) x len(mat2[0])
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            # Elementlərin hasilinin cəmi (dot product)
            dot_product = 0
            for k in range(len(mat2)):
                dot_product += mat1[i][k] * mat2[k][j]
            row.append(dot_product)
        result.append(row)

    return result
