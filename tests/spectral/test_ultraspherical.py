"""Tests for ultraspherical spectral method functions."""

import numpy as np
import pytest

from chebpy.spectral import (
    ultraspherical_bc_row,
    ultraspherical_conversion,
    ultraspherical_diff,
    ultraspherical_multiplication,
    ultraspherical_solve,
)
from chebpy.utilities import Interval


class TestUltrasphericalDiff:
    """Test ultraspherical differentiation matrices."""

    def test_diff_empty(self):
        """Test differentiation with n < 1 returns empty matrix."""
        D = ultraspherical_diff(0, 0)
        assert D.shape == (0, 0)

    def test_diff_chebyshev_first_derivative(self):
        """Test Chebyshev differentiation (lambda=0)."""
        n = 5
        D = ultraspherical_diff(n, 0)
        assert D.shape == (n, n)
        # For Chebyshev: d/dx T_k = k * U_{k-1}
        # D should have superdiagonal [0, 1, 2, 3, 4]
        expected_diag = np.arange(n)
        actual_diag = D.diagonal(1)
        np.testing.assert_array_almost_equal(actual_diag, expected_diag[:-1])

    def test_diff_ultraspherical_general(self):
        """Test general ultraspherical differentiation (lambda > 0)."""
        n = 5
        lmbda = 2
        D = ultraspherical_diff(n, lmbda)
        assert D.shape == (n, n)
        # For lambda > 0: d/dx C^(λ)_k = 2λ * C^(λ+1)_{k-1}
        expected_diag = 2 * lmbda * np.ones(n - 1)
        actual_diag = D.diagonal(1)
        np.testing.assert_array_almost_equal(actual_diag, expected_diag)


class TestUltrasphericalConversion:
    """Test ultraspherical conversion matrices."""

    def test_conversion_empty(self):
        """Test conversion with n < 1."""
        S = ultraspherical_conversion(0, 0)
        assert S.shape == (1, 1)

    def test_conversion_chebyshev_to_c1(self):
        """Test T -> C^(1) conversion."""
        n = 6
        S = ultraspherical_conversion(n, 0)
        assert S.shape == (n, n)
        # Main diagonal: [1, 0.5, 0.5, ...]
        main_diag = S.diagonal(0)
        assert main_diag[0] == 1.0
        assert all(main_diag[1:] == 0.5)
        # Superdiagonal at offset 2: -0.5
        if n >= 3:
            super_diag = S.diagonal(2)
            assert all(super_diag == -0.5)

    def test_conversion_general(self):
        """Test general C^(λ) -> C^(λ+1) conversion."""
        n = 6
        lmbda = 2
        S = ultraspherical_conversion(n, lmbda)
        assert S.shape == (n, n)
        # Main diagonal structure
        main_diag = S.diagonal(0)
        assert main_diag[0] == 1.0
        assert main_diag[1] == pytest.approx(lmbda / (lmbda + 1))
        for k in range(2, n):
            assert main_diag[k] == pytest.approx(lmbda / (lmbda + k))

    def test_conversion_small_n(self):
        """Test conversion with small n values."""
        # n=1
        S = ultraspherical_conversion(1, 0)
        assert S.shape == (1, 1)
        assert S[0, 0] == 1.0

        # n=2 (no superdiagonal)
        S = ultraspherical_conversion(2, 0)
        assert S.shape == (2, 2)

        # n=2 with lambda > 0
        S = ultraspherical_conversion(2, 1)
        assert S.shape == (2, 2)

        # n=4 (with superdiag structure at offset 2)
        S = ultraspherical_conversion(4, 0)
        assert S.shape == (4, 4)

        # n=4 with lambda > 0
        S = ultraspherical_conversion(4, 1)
        assert S.shape == (4, 4)


class TestUltrasphericalMultiplication:
    """Test ultraspherical multiplication matrices."""

    def test_multiplication_constant(self):
        """Test multiplication by constant function."""
        coeffs_f = np.array([3.0])  # f(x) = 3
        n = 4
        M = ultraspherical_multiplication(coeffs_f, n)
        assert M.shape == (n + 1, n + 1)
        # Multiplying by constant 3 should give 3*I
        np.testing.assert_array_almost_equal(M.toarray(), 3 * np.eye(n + 1))

    def test_multiplication_linear(self):
        """Test multiplication by x (T_1)."""
        coeffs_f = np.array([0.0, 1.0])  # f(x) = x = T_1
        n = 4
        M = ultraspherical_multiplication(coeffs_f, n)
        assert M.shape == (n + 1, n + 1)
        # T_1 * T_k = 0.5 * (T_{k+1} + T_{k-1})


class TestUltrasphericalBCRow:
    """Test ultraspherical boundary condition rows."""

    def test_bc_row_dirichlet_left(self):
        """Test Dirichlet BC at left endpoint."""
        n = 5
        interval = Interval(-1, 1)
        row = ultraspherical_bc_row(n, interval, 0, "left")
        assert len(row) == n + 1
        # T_k(-1) = (-1)^k
        expected = np.array([(-1) ** k for k in range(n + 1)])
        np.testing.assert_array_almost_equal(row, expected)

    def test_bc_row_dirichlet_right(self):
        """Test Dirichlet BC at right endpoint."""
        n = 5
        interval = Interval(-1, 1)
        row = ultraspherical_bc_row(n, interval, 0, "right")
        assert len(row) == n + 1
        # T_k(1) = 1 for all k
        expected = np.ones(n + 1)
        np.testing.assert_array_almost_equal(row, expected)

    def test_bc_row_neumann_left(self):
        """Test Neumann BC at left endpoint."""
        n = 5
        interval = Interval(-1, 1)
        row = ultraspherical_bc_row(n, interval, 1, "left")
        assert len(row) == n + 1
        # T'_k(-1) = (-1)^{k+1} * k^2

    def test_bc_row_neumann_right(self):
        """Test Neumann BC at right endpoint."""
        n = 5
        interval = Interval(-1, 1)
        row = ultraspherical_bc_row(n, interval, 1, "right")
        assert len(row) == n + 1
        # T'_k(1) = k^2
        expected = np.array([k**2 for k in range(n + 1)], dtype=float)
        np.testing.assert_array_almost_equal(row, expected)

    def test_bc_row_second_derivative(self):
        """Test second derivative BC."""
        n = 5
        interval = Interval(-1, 1)
        row = ultraspherical_bc_row(n, interval, 2, "right")
        assert len(row) == n + 1

    def test_bc_row_high_order_raises(self):
        """Test that order > 2 raises NotImplementedError."""
        n = 5
        interval = Interval(-1, 1)
        with pytest.raises(NotImplementedError):
            ultraspherical_bc_row(n, interval, 3, "right")

    def test_bc_row_scaled_interval(self):
        """Test BC row with scaled interval."""
        n = 5
        interval = Interval(0, 2)
        row = ultraspherical_bc_row(n, interval, 0, "left")
        assert len(row) == n + 1


class TestUltrasphericalSolve:
    """Test ultraspherical solve function."""

    def test_solve_only_second_order(self):
        """Test that only 2nd order is supported."""
        n = 10
        interval = Interval(0, 1)

        # First order: u' = 0
        coeffs = [0.0, 1.0]
        with pytest.raises(NotImplementedError):
            ultraspherical_solve(coeffs, None, n, interval, 0, 0)

        # Third order: u''' = 0
        coeffs = [0.0, 0.0, 0.0, 1.0]
        with pytest.raises(NotImplementedError):
            ultraspherical_solve(coeffs, None, n, interval, 0, 0)

    def test_solve_no_bcs(self):
        """Test solving with no boundary conditions."""
        coeffs = [0.0, 0.0, 1.0]
        n = 10
        interval = Interval(0, 1)
        rhs_coeffs = np.zeros(n)

        sol_coeffs = ultraspherical_solve(coeffs, rhs_coeffs, n, interval, None, None)
        assert len(sol_coeffs) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
