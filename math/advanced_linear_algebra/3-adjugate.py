#!/usr/bin/env python3
"""
Calculates the adjugate matrix of a matrix
"""


def determinant(matrix):
    """
    Helper function to calculate determinant
    """
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        sub_matrix = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub_matrix)
    return det


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    # 1. Kofaktor matrisini hesablayırıq
    cofactor_matrix = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub_matrix = [row[:j] + row[j + 1:] for row in (matrix[:i] +
                          matrix[i + 1:])]
            minor_val = determinant(sub_matrix)
            row_cofactors.append(((-1) ** (i + j)) * minor_val)
        cofactor_matrix.append(row_cofactors)

    # 2. Transponir edirik (Adjugate matrisi almaq üçün)
    adjugate_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(cofactor_matrix[i][j])
        adjugate_matrix.append(new_row)

    return adjugate_matrix
