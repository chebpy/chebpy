"""Tests for numerical linear algebra helpers."""

import numpy as np
import pytest

from chebpy import fov, polyvalm


def test_polyvalm_linear_identity() -> None:
    """A monic linear polynomial evaluates to the matrix itself."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(polyvalm([1, 0], matrix), matrix)


def test_polyvalm_quadratic() -> None:
    """A quadratic matrix polynomial matches explicit multiplication."""
    matrix = np.array([[0.0, 1.0], [-2.0, 3.0]])
    expected = 2 * (matrix @ matrix) - 3 * matrix + 4 * np.eye(2)
    np.testing.assert_allclose(polyvalm([2, -3, 4], matrix), expected)


def test_polyvalm_rejects_non_square_matrix() -> None:
    """Non-square matrices are rejected."""
    with pytest.raises(ValueError, match="square"):
        polyvalm([1, 0], np.ones((2, 3)))


@pytest.mark.parametrize("coeffs", [[], [[1, 0]]])
def test_polyvalm_rejects_malformed_coefficients(coeffs: object) -> None:
    """Polynomial coefficients must be a non-empty vector."""
    with pytest.raises(ValueError, match="non-empty 1-D"):
        polyvalm(coeffs, np.eye(2))


def test_fov_jordan_block_has_radius_half() -> None:
    """The 2x2 Jordan block has disk field of values with radius 1/2."""
    matrix = np.array([[0.0, 1.0], [0.0, 0.0]])
    boundary = fov(matrix, n=64)
    theta = np.linspace(0.0, 2.0 * np.pi, 129)
    np.testing.assert_allclose(np.abs(boundary(theta)), 0.5, atol=1e-12)


def test_fov_returns_complex_periodic_chebfun() -> None:
    """The boundary is complex-valued and evaluable on [0, 2*pi]."""
    matrix = np.array([[1.0, 2.0], [0.0, 3.0]])
    boundary = fov(matrix, n=64)
    values = boundary(np.array([0.0, np.pi, 2.0 * np.pi]))
    assert boundary.iscomplex
    assert np.iscomplexobj(values)
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[-1], atol=1e-12)


def test_fov_rejects_non_square_matrix() -> None:
    """Non-square matrices are rejected."""
    with pytest.raises(ValueError, match="square"):
        fov(np.ones((2, 3)))
