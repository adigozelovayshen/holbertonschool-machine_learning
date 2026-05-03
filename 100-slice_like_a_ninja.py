#!/usr/bin/env python3
"""Slices a matrix along specific axes using numpy"""
import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    # Matrisin neçə ölçüsü olduğunu tapırıq
    slices = [slice(None)] * len(matrix.shape)

    # Verilmiş lüğətdəki (axes) məlumatlara əsasən slice obyektlərini yerləşdiririk
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)

    # Siyahını tuple-a çevirib matrisə tətbiq edirik
    return matrix[tuple(slices)]
