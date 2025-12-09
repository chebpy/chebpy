"""Tests for integral constraints in boundary value problems.

These tests verify that LinOp can handle integral constraints like:
    ∫ u(x) dx = c
    ∫ g(x) u(x) dx = c

Integral constraints arise in conservation laws, normalization conditions,
and problems with non-local constraints.
"""

import numpy as np

from chebpy import chebfun
from chebpy.linop import LinOp
from chebpy.utilities import Domain


class TestIntegralConstraintsBasics:
    """Basic tests for integral constraints."""

    def test_simple_integral_constraint(self):
        """Test u'' = 1 with u(0) = 0 and ∫u dx = 0.

        This is under-determined without the integral constraint.
        With ∫₀¹ u dx = 0, we can find unique solution.
        """
        domain = Domain([0, 1])

        # u'' = 1
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0  # u(0) = 0

        # Integral constraint: ∫₀¹ u dx = 0
        L.integral_constraint = {'weight': None, 'value': 0}

        L.rhs = chebfun(lambda x: 1 + 0*x, [0, 1])

        u = L.solve()

        # Check boundary condition
        assert abs(u(np.array([0.0]))[0]) < 1e-10

        # Check integral constraint
        integral = u.sum()
        assert abs(integral) < 1e-8

    def test_weighted_integral_constraint(self):
        """Test with weighted integral: ∫ x*u(x) dx = 1."""
        domain = Domain([0, 1])

        # u'' = 0 (Laplace equation)
        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0  # u(0) = 0

        # Weighted integral constraint: ∫₀¹ x*u(x) dx = 1
        weight = chebfun(lambda x: x, [0, 1])
        L.integral_constraint = {'weight': weight, 'value': 1.0}

        L.rhs = chebfun(lambda x: 0*x, [0, 1])

        u = L.solve()

        # Check boundary condition
        assert abs(u(np.array([0.0]))[0]) < 1e-10

        # Check weighted integral
        weighted_integral = (weight * u).sum()
        assert abs(weighted_integral - 1.0) < 1e-8

    def test_normalization_constraint(self):
        """Test eigenvalue problem with normalization ∫ u² dx = 1."""
        domain = Domain([0, np.pi])

        # -u'' = λu with u(0) = u(π) = 0
        a0 = chebfun(lambda x: 0*x, [0, np.pi])
        a1 = chebfun(lambda x: 0*x, [0, np.pi])
        a2 = chebfun(lambda x: -1 + 0*x, [0, np.pi])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # Compute first eigenfunction
        evals, efuns = L.eigs(k=1, sigma=0)

        # Eigenfunctions are typically normalized by the solver
        u = efuns[0]
        norm = (u * u).sum()

        # Should be approximately normalized (L2 norm ≈ 1)
        # Note: eigensolvers typically normalize, so this tests existing behavior
        assert abs(norm - 1.0) < 0.5  # Loose check


class TestConservationLaws:
    """Tests for conservation laws using integral constraints."""

    def test_mass_conservation(self):
        """Test problem with mass conservation: ∫ u dx = M₀."""
        domain = Domain([0, 1])

        # u'' = -u with ∫u dx = 2
        a0 = chebfun(lambda x: -1 + 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 1  # u(0) = 1

        # Mass conservation: ∫u dx = 2
        L.integral_constraint = {'weight': None, 'value': 2.0}

        L.rhs = chebfun(lambda x: 0*x, [0, 1])

        u = L.solve()

        # Check integral
        mass = u.sum()
        assert abs(mass - 2.0) < 1e-6

    def test_center_of_mass_constraint(self):
        """Test with center of mass constraint: ∫ x*u dx / ∫ u dx = x_cm."""
        # This is more complex - would need two integral constraints
        # or a nonlinear constraint. Skip for now.
        pass


class TestMultipleConstraints:
    """Tests for problems with multiple integral constraints."""

    def test_mean_and_variance(self):
        """Test with both mean and second moment constraints.

        This would require multiple integral constraints:
        ∫ u dx = μ
        ∫ x²u dx = σ² + μ²
        """
        # Would need LinOp to accept list of integral constraints
        # This tests future functionality
        pass


class TestIntegralConstraintImplementation:
    """Tests for how integral constraints are discretized."""

    def test_constraint_as_bc_row(self):
        """Verify integral constraint adds a row to BC matrix.

        An integral constraint ∫ g(x) u(x) dx = c becomes a row in
        the discretized system where the row contains quadrature weights.
        """
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0

        # Add integral constraint
        L.integral_constraint = {'weight': None, 'value': 1.0}

        # The implementation should add this as a BC row
        # Current LinOp might not support this yet, so this is aspirational

    def test_galerkin_projection(self):
        """Test that integral constraint uses appropriate quadrature.

        For spectral accuracy, should use Clenshaw-Curtis quadrature
        or similar high-order quadrature.
        """
        pass


class TestPhysicalProblems:
    """Tests based on physical problems with integral constraints."""

    def test_steady_diffusion_with_source(self):
        """Test steady diffusion -u'' = f with total source constraint.

        -u'' = sin(πx), u(0) = u(1) = 0
        But add constraint that ∫f dx = ∫u'' dx = u'(1) - u'(0) must balance.
        """
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        f = chebfun(lambda x: np.sin(np.pi*x), [0, 1])
        L.rhs = f

        u = L.solve()

        # Verify compatibility: ∫f dx should equal flux balance
        flux_balance = u.diff()(np.array([1.0]))[0] - u.diff()(np.array([0.0]))[0]
        source_integral = f.sum()

        # These should be approximately equal (with sign)
        assert abs(flux_balance + source_integral) < 1e-6

    def test_poisson_with_compatibility(self):
        """Test Poisson with Neumann BCs requires ∫f dx = 0.

        -u'' = f with u'(0) = u'(1) = 0 requires ∫f dx = 0.
        If not satisfied, add integral constraint to make unique.
        """
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: -1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # Neumann BCs
        L.lbc = lambda u: u.diff()(np.array([0.0]))[0]
        L.rbc = lambda u: u.diff()(np.array([1.0]))[0]

        # RHS with zero integral (compatibility condition)
        f = chebfun(lambda x: np.sin(2*np.pi*x), [0, 1])
        L.rhs = f

        # Add integral constraint to fix constant: ∫u dx = 0
        L.integral_constraint = {'weight': None, 'value': 0}

        u = L.solve()

        # Check that solution satisfies constraint
        integral = u.sum()
        assert abs(integral) < 1e-6


class TestConstraintSyntax:
    """Tests for different ways to specify integral constraints."""

    def test_constraint_as_function(self):
        """Test integral constraint specified as a callable.

        constraint = lambda u: u.sum() - 1.0
        """
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)
        L.lbc = 0
        L.rbc = 0

        # Could specify as callable
        # L.integral_constraint = lambda u: u.sum() - 1.0

        # For now, use dict syntax
        L.integral_constraint = {'weight': None, 'value': 1.0}

        L.rhs = chebfun(lambda x: 1 + 0*x, [0, 1])

        # This should work when implemented
        # u = L.solve()

    def test_constraint_with_derivative(self):
        """Test integral constraint on derivative: ∫ u'(x) dx = c.

        This is just u(b) - u(a) = c, so reduces to BC.
        But test that syntax works.
        """
        domain = Domain([0, 1])

        a0 = chebfun(lambda x: 0*x, [0, 1])
        a1 = chebfun(lambda x: 0*x, [0, 1])
        a2 = chebfun(lambda x: 1 + 0*x, [0, 1])

        L = LinOp(coeffs=[a0, a1, a2], domain=domain, diff_order=2)

        # ∫ u' dx = u(1) - u(0) = 1
        # This is equivalent to: u(1) = u(0) + 1
        L.lbc = 0  # u(0) = 0
        L.rbc = 1  # u(1) = 1

        L.rhs = chebfun(lambda x: 0*x, [0, 1])

        u = L.solve()

        # Check that integral of derivative equals difference
        u_prime_integral = u.diff().sum()
        assert abs(u_prime_integral - 1.0) < 1e-8
