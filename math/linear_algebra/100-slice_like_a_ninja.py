#!/usr/bin/env python3
"""
Slices a matrix along specific axes
"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes
    """
    # Matrisin ölçüləri qədər bütün elementləri (:) əhatə edən slice siyahısı
    slc = [slice(None)] * len(matrix.shape)

    # Verilən oxları (axis) və onların slice parametrlərini tətbiq edirik
    for axis, params in axes.items():
        slc[axis] = slice(*params)

    # Tuple formatına çevirib matrisi dilimləyirik
    return matrix[tuple(slc)]
