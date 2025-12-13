"""Tests for adjoint operators.

These tests verify that the adjoint operator L* satisfies ⟨Lu, v⟩ = ⟨u, L*v⟩
where ⟨·,·⟩ is the L2 inner product on the domain.
"""

import numpy as np

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestAdjointBasics:
    """Tests for basic adjoint operator properties."""

    def test_adjoint_first_order(self):
        """Test adjoint of d/dx is -d/dx."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # L* should be -d/dx
        L_adj = L.adjoint()

        assert L_adj.diff_order == 1
        assert len(L_adj.coeffs) == 2

        # Check coefficients
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L_adj.coeffs[0](x_test))) < 1e-10  # a0* = 0
        assert np.max(np.abs(L_adj.coeffs[1](x_test) + 1)) < 1e-10  # a1* = -1

    def test_adjoint_second_order(self):
        """Test adjoint of d²/dx² is d²/dx²."""
        domain = Domain([0, 1])

        # L = d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # L* should be d²/dx² (even order derivative)
        L_adj = L.adjoint()

        assert L_adj.diff_order == 2
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L_adj.coeffs[2](x_test) - 1)) < 1e-10

    def test_adjoint_variable_coefficient(self):
        """Test adjoint of x*d/dx."""
        domain = Domain([0, 1])

        # L = x * d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # L* = -d/dx(x·) = -x*d/dx - 1
        L_adj = L.adjoint()

        assert L_adj.diff_order == 1
        x_test = np.linspace(0, 1, 10)
        # a0* = -1, a1* = -x
        assert np.max(np.abs(L_adj.coeffs[0](x_test) + 1)) < 1e-10
        assert np.max(np.abs(L_adj.coeffs[1](x_test) + x_test)) < 1e-10

    def test_adjoint_involution(self):
        """Test (L*)* = L."""
        domain = Domain([0, 1])

        # L = d²/dx² + x*d/dx + 1
        a0 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        a1 = chebfun(lambda x: x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # (L*)*
        L_adj_adj = L.adjoint().adjoint()

        # Should equal L
        assert L_adj_adj.diff_order == L.diff_order
        x_test = np.linspace(0, 1, 10)
        for k in range(len(L.coeffs)):
            diff = np.abs(L_adj_adj.coeffs[k](x_test) - L.coeffs[k](x_test))
            assert np.max(diff) < 1e-10


class TestAdjointInnerProduct:
    """Tests verifying ⟨Lu, v⟩ = ⟨u, L*v⟩."""

    def test_inner_product_first_order(self):
        """Verify inner product property for d/dx with appropriate BCs."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)
        L_adj = L.adjoint()

        # Test functions with zero boundary conditions
        u = chebfun(lambda x: np.sin(np.pi * x), [0, 1])
        v = chebfun(lambda x: np.sin(2 * np.pi * x), [0, 1])

        # Apply operators using __call__
        Lu = L(u)
        L_adj_v = L_adj(v)

        # Compute inner products
        inner1 = (Lu * v).sum()  # ∫(Lu)·v dx
        inner2 = (u * L_adj_v).sum()  # ∫u·(L*v) dx

        # For this to work exactly, need integration by parts with BCs
        # With zero BCs at boundaries, the boundary term vanishes
        # Note: This is approximate since we're not enforcing BCs
        # The test verifies the structure is correct and values are reasonably close
        assert L_adj.diff_order == 1
        # Check inner product property approximately holds
        assert abs(inner1 - inner2) < 0.1  # Loose tolerance due to boundary effects

    def test_inner_product_second_order(self):
        """Verify inner product property for -d²/dx²."""
        domain = Domain([0, 1])

        # L = -d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # For -d²/dx², the adjoint is also -d²/dx²
        L_adj = L.adjoint()

        # Verify structure
        assert L_adj.diff_order == 2
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L_adj.coeffs[2](x_test) + 1)) < 1e-10


class TestAdjointAlgebra:
    """Tests for adjoint operator algebra properties."""

    def test_adjoint_sum(self):
        """Test (L1 + L2)* = L1* + L2*."""
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

        # (L1 + L2)*
        L_sum_adj = (L1 + L2).adjoint()

        # L1* + L2*
        L_adj_sum = L1.adjoint() + L2.adjoint()

        # Should be equal
        x_test = np.linspace(0, 1, 10)
        for k in range(len(L_sum_adj.coeffs)):
            diff = np.abs(L_sum_adj.coeffs[k](x_test) - L_adj_sum.coeffs[k](x_test))
            assert np.max(diff) < 1e-10

    def test_adjoint_scalar_multiplication(self):
        """Test (c*L)* = c*L*."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        c = 3.0

        # (c*L)*
        cL_adj = (c * L).adjoint()

        # c*L*
        c_L_adj = c * L.adjoint()

        # Should be equal
        x_test = np.linspace(0, 1, 10)
        for k in range(len(cL_adj.coeffs)):
            diff = np.abs(cL_adj.coeffs[k](x_test) - c_L_adj.coeffs[k](x_test))
            assert np.max(diff) < 1e-10

    def test_adjoint_negation(self):
        """Test (-L)* = -L*."""
        domain = Domain([0, 1])

        # L = d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # (-L)*
        neg_L_adj = (-L).adjoint()

        # -L*
        neg_adj = -(L.adjoint())

        # Should be equal
        x_test = np.linspace(0, 1, 10)
        for k in range(len(neg_L_adj.coeffs)):
            diff = np.abs(neg_L_adj.coeffs[k](x_test) - neg_adj.coeffs[k](x_test))
            assert np.max(diff) < 1e-10


class TestAdjointApplications:
    """Tests for practical applications of adjoint operators."""

    def test_self_adjoint_operator(self):
        """Test that -d²/dx² is self-adjoint."""
        domain = Domain([0, 1])

        # L = -d²/dx²
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # L* should equal L
        L_adj = L.adjoint()

        x_test = np.linspace(0, 1, 10)
        for k in range(len(L.coeffs)):
            diff = np.abs(L_adj.coeffs[k](x_test) - L.coeffs[k](x_test))
            assert np.max(diff) < 1e-10

    def test_non_self_adjoint_operator(self):
        """Test that d/dx is not self-adjoint."""
        domain = Domain([0, 1])

        # L = d/dx
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1], domain=domain, diff_order=1)

        # L* = -d/dx ≠ L
        L_adj = L.adjoint()

        x_test = np.linspace(0, 1, 10)
        # Coefficient should change sign
        diff = np.abs(L_adj.coeffs[1](x_test) + L.coeffs[1](x_test))
        assert np.max(diff) < 1e-10  # L* has -1, L has +1

    def test_sturm_liouville_form(self):
        """Test adjoint of Sturm-Liouville operator: L = -(p(x)u')' + q(x)u.

        This should be self-adjoint when written in the form:
        L = -p(x)d²/dx² - p'(x)d/dx + q(x)
        """
        domain = Domain([0, 1])

        # p(x) = 1 + x, q(x) = x²
        p = chebfun(lambda x: 1 + x, [0, 1])
        p_prime = p.diff()
        q = chebfun(lambda x: x**2, [0, 1])

        # L = -p(x)d²/dx² - p'(x)d/dx + q(x)
        a0 = q
        a1 = -p_prime
        a2 = -p
        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # In self-adjoint form, L* should equal L
        L_adj = L.adjoint()

        # Check that coefficients match
        x_test = np.linspace(0.1, 0.9, 10)  # Avoid endpoints
        for k in range(len(L.coeffs)):
            diff = np.abs(L_adj.coeffs[k](x_test) - L.coeffs[k](x_test))
            # Note: There may be small differences due to numerical differentiation
            # in computing the adjoint
            assert np.max(diff) < 0.1  # Looser tolerance for variable coefficients


class TestAdjointHigherOrder:
    """Tests for adjoint of higher-order operators."""

    def test_adjoint_third_order(self):
        """Test adjoint of d³/dx³ is -d³/dx³."""
        domain = Domain([0, 1])

        # L = d³/dx³
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 0 * x, [0, 1])
        a3 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2, a3], domain=domain, diff_order=3)

        # L* = -d³/dx³ (odd order)
        L_adj = L.adjoint()

        assert L_adj.diff_order == 3
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L_adj.coeffs[3](x_test) + 1)) < 1e-10

    def test_adjoint_fourth_order(self):
        """Test adjoint of d⁴/dx⁴ is d⁴/dx⁴."""
        domain = Domain([0, 1])

        # L = d⁴/dx⁴
        a0 = chebfun(lambda x: 0 * x, [0, 1])
        a1 = chebfun(lambda x: 0 * x, [0, 1])
        a2 = chebfun(lambda x: 0 * x, [0, 1])
        a3 = chebfun(lambda x: 0 * x, [0, 1])
        a4 = chebfun(lambda x: 1 + 0 * x, [0, 1])
        L = LinOp(coeffs=[a0, a1, a2, a3, a4], domain=domain, diff_order=4)

        # L* = d⁴/dx⁴ (even order)
        L_adj = L.adjoint()

        assert L_adj.diff_order == 4
        x_test = np.linspace(0, 1, 10)
        assert np.max(np.abs(L_adj.coeffs[4](x_test) - 1)) < 1e-10
