"""Tests for rectangularization implementation.

This file tests the rectangularization feature step by step:
1. Rectangular differentiation matrices
2. Rectangular discretization
3. LSMR solver for overdetermined systems
4. Eigenvalue accuracy improvements

The implementation follows Driscoll & Hale (2016) "Rectangular spectral collocation".
"""

import numpy as np
import pytest
from scipy import sparse

from chebpy import chebfun
from chebpy.algorithms import chebpts2
from chebpy.linop import LinOp
from chebpy.op_discretization import OpDiscretization
from chebpy.spectral import diff_matrix, diff_matrix_rectangular
from chebpy.utilities import Domain, Interval


class TestRectangularDifferentiationMatrix:
    """Test rectangular differentiation matrix construction."""

    def test_diff_matrix_rectangular_shape(self):
        """Test that rectangular differentiation matrices have correct shape."""
        n = 8
        m = 16
        interval = Interval(-1, 1)

        # First derivative
        D1 = diff_matrix_rectangular(n, m, interval, order=1)
        assert D1.shape == (m + 1, n + 1), f"Expected (17, 9), got {D1.shape}"

        # Second derivative
        D2 = diff_matrix_rectangular(n, m, interval, order=2)
        assert D2.shape == (m + 1, n + 1), f"Expected (17, 9), got {D2.shape}"

    def test_diff_matrix_rectangular_reduces_to_square(self):
        """Test that rectangular reduces to square when m = n."""
        n = 8
        interval = Interval(-1, 1)

        D_square = diff_matrix(n, interval, order=1)
        D_rect = diff_matrix_rectangular(n, n, interval, order=1)

        # Should be identical
        diff = np.linalg.norm((D_square - D_rect).toarray())
        assert diff < 1e-14, f"Square and rectangular differ by {diff}"

    def test_diff_matrix_rectangular_interpolation(self):
        """Test that rectangular diff matrix correctly differentiates smooth functions."""
        n = 16
        m = 32
        interval = Interval(-1, 1)

        # Create rectangular first derivative matrix
        D1 = diff_matrix_rectangular(n, m, interval, order=1)

        # Test function: sin(πx)

        x_coeff = chebpts2(n + 1)  # n+1 coefficient points
        x_coeff_scaled = interval(x_coeff)
        f_vals = np.sin(np.pi * x_coeff_scaled)

        # Apply rectangular differentiation
        df_vals = D1 @ f_vals

        # Expected: π*cos(πx) at m+1 collocation points
        x_colloc = chebpts2(m + 1)
        x_colloc_scaled = interval(x_colloc)
        expected = np.pi * np.cos(np.pi * x_colloc_scaled)

        error = np.linalg.norm(df_vals - expected, np.inf)
        assert error < 1e-9, f"Differentiation error: {error}"

    def test_diff_matrix_rectangular_rejects_underdetermined(self):
        """Test that m < n raises an error."""
        n = 16
        m = 8  # m < n: underdetermined
        interval = Interval(-1, 1)

        with pytest.raises(ValueError, match="m >= n"):
            diff_matrix_rectangular(n, m, interval, order=1)


class TestRectangularDiscretization:
    """Test rectangular discretization in OpDiscretization."""

    def test_build_discretization_with_rectangularization(self):
        """Test that build_discretization creates rectangular matrices."""
        # Simple second-order operator
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a1 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)  # u''

        linop = LinOp([a0, a1, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        n = 16
        m = 32

        # Build rectangular discretization
        disc = OpDiscretization.build_discretization(linop, n, m=m, rectangularization=True)

        # Check that operator block has rectangular shape
        A_block = disc["blocks"][0]
        assert A_block.shape == (m + 1, n + 1), f"Expected (33, 17), got {A_block.shape}"

        # Check metadata
        assert disc["rectangularization"] is True
        assert disc["m_per_block"][0] == m + 1
        assert disc["n_per_block"][0] == n + 1

    def test_rectangularization_heuristic(self):
        """Test automatic m selection using heuristic."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        test_cases = [
            (8, 16),  # n=8:  m = 2*8 = 16
            (16, 32),  # n=16: m = 2*16 = 32
            (32, 64),  # n=32: m = 2*32 = 64
            (64, 114),  # n=64: m = 2*64 = 128 > 64+50, so m = 114
        ]

        for n, expected_m in test_cases:
            disc = OpDiscretization.build_discretization(linop, n, rectangularization=True)
            actual_m = disc["m_per_block"][0] - 1
            assert actual_m == expected_m, f"For n={n}, expected m={expected_m}, got {actual_m}"


class TestLSMRSolver:
    """Test LSMR solver for overdetermined systems."""

    def test_solve_overdetermined_simple(self):
        """Test that LSMR solver handles overdetermined systems."""
        # Create simple overdetermined system: m=20, n=10

        np.random.seed(42)

        m, n = 20, 10
        # Make A well-conditioned by using more structure
        A = sparse.eye(m, n, format="csr") + 0.1 * sparse.random(m, n, density=0.3, format="csr")
        x_true = np.random.randn(n)
        b = A @ x_true

        # Create dummy LinOp for solver access
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.ones_like(x), interval)
        linop = LinOp([a0], domain, diff_order=0)

        # Solve using solve_linear_system (should detect overdetermined and use LSMR)
        x_solved = linop.solve_linear_system(A, b)

        # Should recover true solution
        error = np.linalg.norm(x_solved - x_true)
        assert error < 1e-8, f"LSMR solution error: {error}"

    def test_solve_square_system_unchanged(self):
        """Test that square systems still use LU decomposition."""
        np.random.seed(42)

        n = 10
        A = sparse.random(n, n, density=0.8, format="csr") + sparse.eye(n)
        x_true = np.random.randn(n)
        b = A @ x_true

        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.ones_like(x), interval)
        linop = LinOp([a0], domain, diff_order=0)

        x_solved = linop.solve_linear_system(A, b)

        error = np.linalg.norm(x_solved - x_true)
        assert error < 1e-10, f"LU solution error: {error}"


class TestEigenvalueAccuracy:
    """Test eigenvalue accuracy improvements with rectangularization."""

    def test_basic_eigenvalue_problem(self):
        """Test standard eigenvalue problem: -u'' = λu on [0, π].

        Exact eigenvalues: λ_k = k^2 for k = 1, 2, 3, ...
        """
        domain = Domain([0, np.pi])
        interval = Interval(0, np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a1 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: -np.ones_like(x), interval)  # -u''

        linop = LinOp([a0, a1, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        exact_evals = np.array([k**2 for k in range(1, 6)], dtype=float)

        # Compute eigenvalues with both methods
        evals_square, _ = linop.eigs(k=5, rectangularization=False)
        evals_rect, _ = linop.eigs(k=5, rectangularization=True)

        err_square = np.abs(evals_square - exact_evals)
        err_rect = np.abs(evals_rect - exact_evals)

        # Both should achieve good accuracy
        assert np.max(err_square) < 1e-10, f"Square max error: {np.max(err_square)}"
        assert np.max(err_rect) < 1e-10, f"Rectangular max error: {np.max(err_rect)}"

    def test_rectangular_eigenvalues_are_accurate(self):
        """Test that rectangularization achieves high accuracy for simple problem."""
        domain = Domain([0, np.pi])
        interval = Interval(0, np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: -np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        evals, efuns = linop.eigs(k=5, rectangularization=True)

        # Exact: k^2 for k = 1, 2, 3, 4, 5
        exact = np.array([1, 4, 9, 16, 25], dtype=float)
        rel_errors = np.abs(evals - exact) / exact

        # Spectral methods should achieve high accuracy
        assert np.all(rel_errors < 1e-10), f"Max relative error: {np.max(rel_errors):.2e}"

    def test_eigenfunctions_satisfy_bcs(self):
        """Test that eigenfunctions satisfy boundary conditions.

        This test verifies BCs are satisfied regardless of discretization method.
        """
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: -np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        evals, efuns = linop.eigs(k=3, rectangularization=True)

        for i, ef in enumerate(efuns):
            left_val = ef(0.0)
            right_val = ef(1.0)
            # Machine precision boundary condition satisfaction
            assert abs(left_val) < 1e-12, f"Eigenfunction {i} left BC: {left_val}"
            assert abs(right_val) < 1e-12, f"Eigenfunction {i} right BC: {right_val}"


class TestBackwardCompatibility:
    """Test that rectangularization=False maintains existing behavior."""

    def test_default_is_square(self):
        """Test that default behavior is square discretization."""
        domain = Domain([0, 1])
        interval = Interval(0, 1)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        n = 16

        # Default: no rectangularization
        disc = OpDiscretization.build_discretization(linop, n)

        # Should be square
        assert disc["rectangularization"] is False
        A_block = disc["blocks"][0]
        assert A_block.shape[0] == A_block.shape[1], "Default should be square"

    def test_eigs_default_is_square(self):
        """Test that eigs() default is square discretization."""
        domain = Domain([0, np.pi])
        interval = Interval(0, np.pi)
        a0 = chebfun(lambda x: np.zeros_like(x), interval)
        a2 = chebfun(lambda x: -np.ones_like(x), interval)

        linop = LinOp([a0, None, a2], domain, diff_order=2)
        linop.lbc = 0.0
        linop.rbc = 0.0

        # Default call (no rectangularization parameter)
        evals, efuns = linop.eigs(k=3)

        # Should work and return 3 eigenvalues
        assert len(evals) == 3
        assert len(efuns) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
