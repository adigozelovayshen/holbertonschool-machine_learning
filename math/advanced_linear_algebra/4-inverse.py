#!/usr/bin/env python3
"""
Calculates the inverse of a matrix
"""


def determinant(matrix):
    """ Helper to calculate determinant """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Returns None if matrix is singular
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Determinant hesablanır
    det = determinant(matrix)
    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    # 1. Kofaktor matrisi və Transponir (Adjugate)
    adj = []
    for j in range(n):
        adj_row = []
        for i in range(n):
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] +
                          matrix[i + 1:])]
            # Kofaktorun transponiri birbaşa burada hesablanır (i və j yerini dəyişərək)
            cofactor = ((-1) ** (i + j)) * determinant(sub_matrix)
            adj_row.append(cofactor / det)
        adj.append(adj_row)

    return adj
