"""Sparse matrix and numerical utility functions."""

import numpy as np
from scipy import sparse


def extract_scalar(value, negate: bool = False):
    """Extract scalar value from various types (array, list, scalar, AdChebfunScalar).

    Unified extraction for residuals and BC values that may be wrapped in
    different container types.

    Args:
        value: Value to extract (scalar, list, array, AdChebfunScalar, etc.)
        negate: If True, negate the extracted value

    Returns:
        Extracted float value
    """
    # Handle AdChebfunScalar wrapping (avoid circular import by checking type name)
    if hasattr(value, "value") and type(value).__name__ == "AdChebfunScalar":
        value = value.value

    # Extract from array/list
    if isinstance(value, (list, np.ndarray)):
        scalar = float(value[0] if len(value) > 0 else 0.0)
    elif np.isscalar(value):
        scalar = float(value)
    else:
        scalar = 0.0

    return -scalar if negate else scalar


def jacobian_to_row(jac):
    """Convert Jacobian (sparse or dense) to a dense row vector.

    Args:
        jac: Jacobian matrix (sparse CSR/CSC/LIL or dense array)

    Returns:
        1D numpy array
    """
    if sparse.issparse(jac):
        return jac.toarray().ravel()
    else:
        return np.atleast_1d(jac).ravel()


def is_nearly_zero(mat, threshold: float = 1e-12) -> bool:
    """Check if matrix/row is essentially zero.

    Args:
        mat: Matrix or row vector (array or sparse matrix)
        threshold: Threshold below which values are considered zero

    Returns:
        True if norm is below threshold
    """
    if sparse.issparse(mat):
        return np.linalg.norm(mat.toarray()) < threshold
    return np.linalg.norm(mat) < threshold


def prune_sparse(mat: sparse.spmatrix, threshold: float = 1e-14) -> sparse.spmatrix:
    """Prune tiny coefficients from sparse matrix for numerical stability.

    Args:
        mat: Sparse matrix (any format)
        threshold: Values below this magnitude are set to zero

    Returns:
        Pruned sparse matrix in CSR format
    """
    mat = mat.tocsr()
    mat.data[np.abs(mat.data) < threshold] = 0
    mat.eliminate_zeros()
    return mat


def sparse_to_dense(mat):
    """Convert sparse matrix to dense array.

    Args:
        mat: Sparse or dense matrix

    Returns:
        Dense numpy array
    """
    if sparse.issparse(mat):
        return mat.toarray()
    return np.asarray(mat)
