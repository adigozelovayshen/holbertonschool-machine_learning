# Dimensionality Reduction

This project covers techniques for dimensionality reduction in Machine Learning, starting with Principal Component Analysis (PCA).

## Tasks

### 0. PCA
Function `pca(X, var=0.95)` that performs Principal Component Analysis on a dataset.
- **X**: `numpy.ndarray` of shape `(n, d)` with zero mean across all dimensions.
- **var**: Fraction of variance to retain.
- **Returns**: Weight matrix `W` of shape `(d, nd)`.
