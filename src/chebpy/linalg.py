"""Numerical linear algebra helpers for ChebPy examples."""

from typing import Any

import numpy as np

from .bndfun import Bndfun
from .chebfun import Chebfun
from .trigtech import Trigtech
from .utilities import Interval


def _as_square_matrix(matrix: Any) -> np.ndarray:
    """Return *matrix* as a 2-D square array."""
    array = np.asarray(matrix)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = "matrix must be a square 2-D array"
        raise ValueError(msg)
    return array


def fov(matrix: Any, n: int = 256) -> Chebfun:
    """Return the boundary of the field of values of a square matrix.

    The returned object is a complex-valued Chebfun backed by
    :class:`~chebpy.trigtech.Trigtech` on ``[0, 2*pi]``.  It parameterizes the
    boundary using the standard rotated-Hermitian eigenvalue method: for each
    angle ``theta``, the eigenvector associated with the largest eigenvalue of
    ``(exp(1j*theta) A + exp(-1j*theta) A*) / 2`` gives the boundary point
    ``v* A v``.

    Args:
        matrix: Square matrix.
        n: Number of equally spaced angles used for the periodic representation.

    Returns:
        Chebfun: Complex-valued boundary parameterization on ``[0, 2*pi]``.

    Raises:
        ValueError: If *matrix* is not square or *n* is less than 2.
    """
    array = _as_square_matrix(matrix)
    n = int(n)
    if n < 2:
        msg = "n must be at least 2"
        raise ValueError(msg)

    values = np.empty(n, dtype=complex)
    for k, theta in enumerate(2.0 * np.pi * np.arange(n) / n):
        rotated = np.exp(1j * theta) * array
        hermitian = 0.5 * (rotated + rotated.conj().T)
        _, eigenvectors = np.linalg.eigh(hermitian)
        vector = eigenvectors[:, -1]
        values[k] = np.vdot(vector, array @ vector)

    interval = Interval(0.0, 2.0 * np.pi)
    onefun = Trigtech.initvalues(values, interval=interval)
    return Chebfun([Bndfun(onefun, interval)])


def polyvalm(coeffs: Any, matrix: Any) -> np.ndarray:
    """Evaluate a polynomial at a square matrix by Horner multiplication.

    Coefficients are ordered highest-degree first, matching :func:`numpy.polyval`.
    For example, ``polyvalm([2, 3, 4], A)`` computes ``2*A@A + 3*A + 4*I``.

    Args:
        coeffs: One-dimensional, non-empty coefficient array ordered from highest
            degree to constant term.
        matrix: Square matrix at which to evaluate the polynomial.

    Returns:
        numpy.ndarray: Matrix polynomial value.

    Raises:
        ValueError: If *coeffs* is not a non-empty 1-D array or *matrix* is not square.
    """
    array = _as_square_matrix(matrix)
    coeff_array = np.asarray(coeffs)
    if coeff_array.ndim != 1 or coeff_array.size == 0:
        msg = "coeffs must be a non-empty 1-D array"
        raise ValueError(msg)

    dtype = np.result_type(coeff_array, array)
    identity = np.eye(array.shape[0], dtype=dtype)
    result = coeff_array[0] * identity
    matrix_array = array.astype(dtype, copy=False)
    for coeff in coeff_array[1:]:
        result = result @ matrix_array + coeff * identity
    return result
