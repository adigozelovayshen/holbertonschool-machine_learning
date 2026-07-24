#!/usr/bin/env python3
"""Normalization Constants"""
import numpy as np


def normalization_constants(X):
    """Calculates the mean and standard deviation of a data set"""
    return np.mean(X, axis=0), np.std(X, axis=0)
