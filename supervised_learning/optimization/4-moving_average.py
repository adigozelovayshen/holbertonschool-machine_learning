#!/usr/bin/env python3
"""Moving Average"""


def moving_average(data, beta):
    """Calculates the exponentially weighted moving average of a data set"""
    v = 0
    v_list = []
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        v_list.append(v / (1 - beta ** (i + 1)))
    return v_list
