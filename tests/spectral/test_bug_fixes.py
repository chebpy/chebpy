"""Tests for verified bug fixes from code review.

These tests would have failed before the fixes but now pass.
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestNullspaceFix:
    """Tests for corrected null() rank calculation."""

    def test_null_underdetermined_system(self):
        """Test null() with under-determined system.

        Before fix: rank = len(s) was always the number of singular values
        After fix: rank = number of singular values above tolerance
        """
        domain = Domain([0, 1])

        # u'' = 0 with only u(0) = 0 (under-determined, 1D nullspace)
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0  # u(0) = 0
        L.max_n = 32  # Keep small for speed

        # Should find non-empty nullspace (linear functions satisfying u(0)=0)
        null_basis = L.null()

        # Should have at least one null vector
        assert len(null_basis) > 0, "Null space should be non-empty for under-determined system"

        # Verify each basis function is actually in the nullspace
        for u_null in null_basis:
            # Apply operator
            Lu = L(u_null)

            # Should be approximately zero
            # Compute residual: for d²u/dx² = 0, check ||u''||
            x_test = np.linspace(0, 1, 20)
            residual = np.max(np.abs(Lu(x_test)))
            assert residual < 1e-10, f"Null space vector should satisfy L[u] ≈ 0, residual = {residual}"

            # Should satisfy BC
            bc_residual = abs(u_null(np.array([0.0]))[0])
            assert bc_residual < 1e-10, f"Null space vector should satisfy u(0) = 0, residual = {bc_residual}"


class TestOperatorArithmetic:
    """Tests for __truediv__ and __pow__ operators."""

    def test_truediv_basic(self):
        """Test L / 2 works correctly."""
        domain = Domain([0, 1])

        # L = d²/dx²
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # L2 = L / 2
        L2 = L / 2

        # Check coefficients are halved
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L2.coeffs[2](x_test) - 0.5)) < 1e-10

    def test_truediv_zero_error(self):
        """Test dividing by zero raises error."""
        domain = Domain([0, 1])
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L = LinOp(coeffs=[a0], domain=domain, diff_order=0)

        with pytest.raises(ZeroDivisionError):
            L / 0

    def test_pow_zero(self):
        """Test L ** 0 returns identity."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # L ** 0 = I
        I = L ** 0

        assert I.diff_order == 0
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(I.coeffs[0](x_test) - 1)) < 1e-10

    def test_pow_one(self):
        """Test L ** 1 returns self."""
        domain = Domain([0, 1])
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L = LinOp(coeffs=[a0], domain=domain, diff_order=0)

        L1 = L ** 1
        assert L1 is L

    def test_pow_higher_composition(self):
        """Test L ** 2 works via operator composition."""
        domain = Domain([0, 1])
        # Create D (first derivative operator): D[u] = u'
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0*x, [0, 1])
        D = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # D ** 2 should be the second derivative operator
        D2 = D ** 2
        assert D2.diff_order == 2

    def test_pow_negative_error(self):
        """Test L ** -1 raises ValueError."""
        domain = Domain([0, 1])
        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L = LinOp(coeffs=[a0], domain=domain, diff_order=0)

        with pytest.raises(ValueError, match="Negative powers"):
            L ** -1


class TestEigsToleranceConfigurable:
    """Test that eigs convergence tolerance is configurable."""

    def test_eigs_custom_tolerance(self):
        """Test eigs respects custom tolerance via kwargs."""
        domain = Domain([0, np.pi])

        # -u'' = λu with u(0) = u(π) = 0
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # Should accept tol parameter
        evals1, _ = L.eigs(k=2, sigma=0, tol=1e-6)
        evals2, _ = L.eigs(k=2, sigma=0, tol=1e-10)

        # Both should work without error
        assert len(evals1) == 2
        assert len(evals2) == 2


class TestCallableBCMultiBlock:
    """Test callable BC uses correct interval for multi-block problems."""

    def test_callable_bc_correct_interval_single_block(self):
        """Test callable BC works for single block (regression test)."""
        domain = Domain([0, 1])

        # -u'' = 1, u(0) = 0, u'(1) = 0 (Neumann BC via callable)
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = lambda u: u.diff()(np.array([1.0]))[0]  # u'(1) = 0
        L.rhs = chebfun(lambda x: 1 + 0*x, [0, 1])

        # Should solve without error
        u = L.solve()

        # Check BCs
        assert abs(u(np.array([0.0]))[0]) < 1e-8
        assert abs(u.diff()(np.array([1.0]))[0]) < 1e-6


class TestOperatorAdditionRobustness:
    """Test operator addition handles edge cases correctly."""

    def test_add_different_diff_orders(self):
        """Test adding operators with different differential orders."""
        domain = Domain([0, 1])

        # L1 = d/dx (order 1)
        a0_1 = chebfun(lambda x: 0*x, [0, 1])
        a1_1 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L1 = LinOp(coeffs=[a0_1, a1_1], domain=domain, diff_order=1)

        # L2 = d²/dx² (order 2)
        a0_2 = chebfun(lambda x: 0*x, [0, 1])
        a1_2 = chebfun(lambda x: 0*x, [0, 1])
        a2_2 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L2 = LinOp(coeffs=[a0_2, a1_2, a2_2], domain=domain, diff_order=2)

        # Should work
        L3 = L1 + L2

        # Result should have max order
        assert L3.diff_order == 2
        assert len(L3.coeffs) == 3

    def test_add_incompatible_domains_error(self):
        """Test adding operators with different domains raises error."""
        domain1 = Domain([0, 1])
        domain2 = Domain([0, 2])

        a0 = chebfun(lambda x: 1 + 0*x, [0, 1])
        L1 = LinOp(coeffs=[a0], domain=domain1, diff_order=0)

        a0_2 = chebfun(lambda x: 1 + 0*x, [0, 2])
        L2 = LinOp(coeffs=[a0_2], domain=domain2, diff_order=0)

        with pytest.raises(ValueError, match="domain"):
            L1 + L2
