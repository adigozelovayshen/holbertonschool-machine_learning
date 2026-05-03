#!/usr/bin/env python3
"""
Slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes
    """
    # Qlobal import qadağan olduğu üçün daxili çağırırıq
    import numpy as np

    # Bütün ölçülər üçün tam slice (:) yaradırıq
    slc = [slice(None)] * len(matrix.shape)

    # Verilmiş oxlar üzrə slice-ları yerləşdiririk
    for axis, params in axes.items():
        slc[axis] = slice(*params)

    # Nəticəni yeni numpy.ndarray kimi qaytarırıq
    return matrix[tuple(slc)]
