#!/usr/bin/env python3
"""Normalize"""
import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix X with mean m and standard deviation s"""
    return (X - m) / s
