"""Tests for operator algebra operations.

These tests verify that operators can be combined algebraically using
addition, scalar multiplication, and other operations.
"""

import numpy as np
import pytest

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestOperatorAddition:
    """Tests for operator addition (L1 + L2)."""

    def test_add_two_operators(self):
        """Test (d/dx + d²/dx²)[u] works correctly."""
        domain = Domain([0, 1])

        # L1 = d/dx
        a0_1 = chebfun(lambda x: 0 * x, [0, 1])
        a1_1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L1 = LinOp(coeffs=[a0_1, a1_1], domain=domain, diff_order=1)

        # L2 = d²/dx²
        a0_2 = chebfun(lambda x: 0 * x, [0, 1])
        a1_2 = chebfun(lambda x: 0 * x, [0, 1])
        a2_2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L2 = LinOp(coeffs=[a0_2, a1_2, a2_2], domain=domain, diff_order=2)

        # L3 = L1 + L2 = d/dx + d²/dx²
        L3 = L1 + L2

        # Check that L3 has correct differential order (max of the two)
        assert L3.diff_order == 2

        # Check that L3 has correct number of coefficients
        assert len(L3.coeffs) == 3

        # Verify coefficients: should be [0, 1, 1]
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L3.coeffs[0](x_test))) < 1e-10  # a0 = 0
        assert np.max(np.abs(L3.coeffs[1](x_test) - 1)) < 1e-10  # a1 = 1
        assert np.max(np.abs(L3.coeffs[2](x_test) - 1)) < 1e-10  # a2 = 1

    def test_add_same_order_operators(self):
        """Test adding two 2nd order operators."""
        domain = Domain([0, 1])

        # L1 = d²/dx²
        a0_1 = chebfun(lambda x: 0 * x, [0, 1])
        a1_1 = chebfun(lambda x: 0 * x, [0, 1])
        a2_1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L1 = LinOp(coeffs=[a0_1, a1_1, a2_1], domain=domain, diff_order=2)

        # L2 = x * I (multiplication operator)
        a0_2 = chebfun(lambda x: x, [0, 1])
        L2 = LinOp(coeffs=[a0_2], domain=domain, diff_order=0)

        # L3 = L1 + L2 = d²/dx² + x*I
        L3 = L1 + L2

        assert L3.diff_order == 2
        assert len(L3.coeffs) == 3

        # Verify coefficients: should be [x, 0, 1]
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L3.coeffs[0](x_test) - x_test)) < 1e-10
        assert np.max(np.abs(L3.coeffs[1](x_test))) < 1e-10
        assert np.max(np.abs(L3.coeffs[2](x_test) - 1)) < 1e-10


class TestScalarMultiplication:
    """Tests for scalar multiplication of operators."""

    def test_multiply_by_scalar(self):
        """Test c * L where c is a scalar."""
        domain = Domain([0, 1])

        # L = d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # L2 = 3 * L = 3 * d²/dx²
        L2 = 3 * L

        assert L2.diff_order == 2
        assert len(L2.coeffs) == 3

        # Check coefficients are scaled
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L2.coeffs[2](x_test) - 3)) < 1e-10

    def test_right_scalar_multiplication(self):
        """Test L * c where c is a scalar."""
        domain = Domain([0, 1])

        # L = d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # L2 = L * 2 = 2 * d²/dx²
        L2 = L * 2

        assert L2.diff_order == 2
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L2.coeffs[2](x_test) - 2)) < 1e-10

    def test_negative_operator(self):
        """Test -L creates negated operator."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # L2 = -L
        L2 = -L

        assert L2.diff_order == 1
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L2.coeffs[1](x_test) + 1)) < 1e-10


class TestOperatorSubtraction:
    """Tests for operator subtraction (L1 - L2)."""

    def test_subtract_operators(self):
        """Test L1 - L2 = L1 + (-L2)."""
        domain = Domain([0, 1])

        # L1 = d²/dx²
        a0_1 = chebfun(lambda x: 0 * x, [0, 1])
        a1_1 = chebfun(lambda x: 0 * x, [0, 1])
        a2_1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L1 = LinOp(coeffs=[a0_1, a1_1, a2_1], domain=domain, diff_order=2)

        # L2 = I (identity)
        a0_2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L2 = LinOp(coeffs=[a0_2], domain=domain, diff_order=0)

        # L3 = L1 - L2 = d²/dx² - I
        L3 = L1 - L2

        assert L3.diff_order == 2
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L3.coeffs[0](x_test) + 1)) < 1e-10  # -1 from -L2
        assert np.max(np.abs(L3.coeffs[2](x_test) - 1)) < 1e-10  # 1 from L1


class TestOperatorAlgebraApplications:
    """Tests for using operator algebra in actual problems."""

    def test_helmholtz_operator(self):
        """Test constructing Helmholtz operator: -d²/dx² - k²I."""
        k = 2.0
        domain = Domain([0, np.pi])

        # Start with -d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, np.pi])
        a1 = chebfun(lambda x: 0 * x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0 * x, [0, np.pi])
        L_laplace = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # Create k²I
        a_id = chebfun(lambda x: k**2 + 0 * x, [0, np.pi])
        L_id = LinOp(coeffs=[a_id], domain=domain, diff_order=0)

        # Helmholtz: L = -d²/dx² - k²I
        L = L_laplace - L_id

        # Set BCs and solve
        L.lbc = 0
        L.rbc = 0
        L.rhs = chebfun(lambda x: np.sin(x), [0, np.pi])

        u = L.solve()

        # Check solution satisfies BCs
        assert abs(u(np.array([0.0]))[0]) < 1e-10
        assert abs(u(np.array([np.pi]))[0]) < 1e-10


class TestOperatorAlgebraEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_incompatible_domains(self):
        """Test that adding operators with different domains raises error."""
        domain1 = Domain([0, 1])
        domain2 = Domain([0, 2])

        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L1 = LinOp(coeffs=[a0, a1], domain=domain1, diff_order=1)

        a0_2 = chebfun(lambda x: 0 * x, [0, 2])
        a1_2 = chebfun(lambda x: 1 + 0 * x, [0, 2])
        L2 = LinOp(coeffs=[a0_2, a1_2], domain=domain2, diff_order=1)

        with pytest.raises(ValueError, match="domain"):
            L1 + L2

    def test_multiply_by_zero(self):
        """Test that 0 * L creates zero operator."""
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        L2 = 0 * L

        x_test = np.linspace(0, 1, 10)
        for coeff in L2.coeffs:
            assert np.max(np.abs(coeff(x_test))) < 1e-10
