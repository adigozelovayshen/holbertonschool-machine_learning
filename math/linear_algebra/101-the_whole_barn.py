#!/usr/bin/env python3
"""
Adds two matrices of any dimension
"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices recursively
    Returns None if shapes are not the same
    """
    # Ölçüləri yoxlayırıq (Şəkil eyni deyilsə None qaytarırıq)
    if len(mat1) != len(mat2):
        return None

    # Əgər elementlər siyahı deyilsə (artıq ən dərin qatdayıqsa) toplayırıq
    if not isinstance(mat1[0], list):
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    # Rekursiv olaraq hər bir alt-matrisi toplayırıq
    res = []
    for i in range(len(mat1)):
        sub_matrix = add_matrices(mat1[i], mat2[i])
        # Əgər alt-matrislərin şəkli fərqli çıxsa (None qayıtsa)
        if sub_matrix is None:
            return None
        res.append(sub_matrix)

    return res
