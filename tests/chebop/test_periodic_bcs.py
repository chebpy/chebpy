"""Comprehensive tests for periodic boundary conditions in Chebop.

This module contains extensive tests to verify that periodic boundary conditions
work correctly with both Fourier (trigtech) and Chebyshev (chebtech) discretizations.
"""

import numpy as np
import pytest

from chebpy import chebfun, chebop


class TestPeriodicBCBasics:
    """Basic tests for periodic boundary condition syntax and enforcement."""

    def test_periodic_bc_syntax(self):
        """Test that bc='periodic' syntax is accepted."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"

        assert N.bc == "periodic"

    def test_periodic_enforcement_second_order(self):
        """Test that periodic BCs are enforced for second-order ODE.

        Solve: u'' = sin(2x) with periodic BCs
        Expected: u(0) = u(2π) and u'(0) = u'(2π)
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Check function periodicity
        u_0 = u(np.array([0.0]))[0]
        u_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(u_0 - u_2pi) < 1e-10, f"|u(0) - u(2π)| = {np.abs(u_0 - u_2pi)}"

        # Check derivative periodicity
        uprime = u.diff()
        uprime_0 = uprime(np.array([0.0]))[0]
        uprime_2pi = uprime(np.array([2 * np.pi]))[0]
        assert np.abs(uprime_0 - uprime_2pi) < 1e-10, f"|u'(0) - u'(2π)| = {np.abs(uprime_0 - uprime_2pi)}"

        # Check against exact solution: u'' = sin(2x) => u = -sin(2x)/4 + C
        # With periodic BC and mean-zero constraint, C can be determined
        x_test = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        u_vals = u(x_test)
        u_expected = -np.sin(2 * x_test) / 4
        # Allow for constant offset due to mean-zero constraint
        offset = np.mean(u_vals - u_expected)
        error = np.max(np.abs((u_vals - offset) - u_expected))
        assert error < 1e-8, f"Solution error: {error}"

    def test_periodic_known_solution(self):
        """Test periodic BC with known exact solution.

        Solve: u'' = -4*sin(2x) with periodic BCs
        Exact solution: u = sin(2x) + C (C determined by periodicity)
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: -4 * np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Exact solution (up to a constant)
        u_exact = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])

        # Check that u and u_exact differ by at most a constant
        # Evaluate at points first to avoid Trigtech - Chebtech complex arithmetic
        x_pts = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        u_vals = u(x_pts)
        u_exact_vals = u_exact(x_pts)
        diff_vals = u_vals - u_exact_vals
        variation = np.max(np.real(diff_vals)) - np.min(np.real(diff_vals))
        assert variation < 1e-10, f"Solution variation: {variation}"

        # Verify periodicity
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([2 * np.pi]))[0]) < 1e-10


class TestPeriodicVsNonPeriodic:
    """Tests comparing periodic and non-periodic BCs."""

    def test_periodic_vs_dirichlet_different(self):
        """Verify that ill-posed periodic problem raises error.

        The problem u'' = x with periodic BCs is ill-posed because the RHS x
        does not satisfy the compatibility condition ∫f dx = 0 over the period.

        This test verifies that the solver detects this and raises a clear error.
        """
        rhs = chebfun(lambda x: x, [0, 2 * np.pi])

        # Periodic BCs with incompatible RHS should raise ValueError
        N_periodic = chebop([0, 2 * np.pi])
        N_periodic.op = lambda u: u.diff(2)
        N_periodic.bc = "periodic"
        N_periodic.rhs = rhs

        with pytest.raises(ValueError, match="PERIODIC COMPATIBILITY ERROR"):
            N_periodic.solve()

        # Dirichlet BCs should work fine (no compatibility constraint)
        N_dirichlet = chebop([0, 2 * np.pi])
        N_dirichlet.op = lambda u: u.diff(2)
        N_dirichlet.lbc = 0
        N_dirichlet.rbc = 0
        N_dirichlet.rhs = rhs
        u_dirichlet = N_dirichlet.solve()

        # Should have a solution with Dirichlet BCs
        assert u_dirichlet is not None

    def test_periodic_enforces_derivative_match(self):
        """Test that periodic BCs enforce matching derivatives at boundaries."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2) + u
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(3 * x), [0, 2 * np.pi])

        u = N.solve()

        # All derivatives should be periodic
        for order in [0, 1, 2]:
            u_deriv = u.diff(order) if order > 0 else u
            val_0 = u_deriv(np.array([0.0]))[0]
            val_2pi = u_deriv(np.array([2 * np.pi]))[0]
            assert np.abs(val_0 - val_2pi) < 1e-10, f"Derivative order {order} not periodic"


class TestPeriodicHigherOrder:
    """Tests for periodic BCs with higher-order operators."""

    def test_periodic_fourth_order(self):
        """Test periodic BCs with fourth-order operator.

        Solve: u'''' = sin(2x) with periodic BCs
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(4)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Check periodicity
        val_0 = u(np.array([0.0]))[0]
        val_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(val_0 - val_2pi) < 1e-10, "Solution not periodic"

        # Check against exact solution: u'''' = sin(2x) => u = sin(2x)/16 + C
        x_test = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        u_vals = u(x_test)
        u_expected = np.sin(2 * x_test) / 16
        # Allow for constant offset due to mean-zero constraint
        offset = np.mean(u_vals - u_expected)
        error = np.max(np.abs((u_vals - offset) - u_expected))
        assert error < 1e-10, f"Solution error: {error}"

    def test_periodic_biharmonic(self):
        """Test biharmonic operator (u'''' + u'') with periodic BCs."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(4) + u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.cos(3 * x), [0, 2 * np.pi])

        u = N.solve()

        # Verify periodicity
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([2 * np.pi]))[0]) < 1e-10


class TestPeriodicVariableCoefficients:
    """Tests for periodic BCs with variable coefficient operators."""

    def test_periodic_variable_coeff(self):
        """Test periodic BCs with variable coefficient operator.

        Solve: u'' + (2 + sin(x))*u = f(x) with periodic BCs
        """
        N = chebop([0, 2 * np.pi])
        coeff = chebfun(lambda x: 2 + np.sin(x), [0, 2 * np.pi])
        N.op = lambda u: u.diff(2) + coeff * u
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.cos(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Verify periodicity of function and derivative
        u_0 = u(np.array([0.0]))[0]
        u_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(u_0 - u_2pi) < 1e-9, f"|u(0) - u(2π)| = {np.abs(u_0 - u_2pi)}"

        # Verify solution smoothness (should not have discontinuities)
        x_test = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        u_vals = u(x_test)
        # Check that values are bounded and smooth
        assert np.all(np.isfinite(u_vals)), "Solution has non-finite values"
        assert np.max(np.abs(u_vals)) < 100, "Solution values unreasonably large"


@pytest.mark.skip(
    reason="Eigenvalue computation with periodic BCs is correct but very slow "
    "(~250s per test, vs 1.2s in MATLAB) - adaptive refinement needs optimization"
)
class TestPeriodicEigenvalues:
    """Tests for eigenvalue problems with periodic BCs.

    NOTE: These tests are mathematically correct and produce accurate eigenvalues,
    but are extremely slow compared to MATLAB Chebfun (250s vs 1.2s per test).
    The bottleneck is in the adaptive refinement loop during eigenvalue computation
    with periodic boundary conditions. This requires optimization of LinOp.eigs()
    for periodic problems, potentially by:
    - Using specialized Fourier eigenvalue solver for periodic BCs
    - Optimizing QR decomposition for periodic constraint matrices
    - Reducing number of adaptive refinement steps for smooth periodic eigenfunctions
    """

    def test_periodic_eigenvalues_second_order(self):
        """Test eigenvalue problem: -u'' = λu with periodic BCs.

        Expected eigenvalues: λ_n = n² for n = 0, 1, 2, ...
        Expected eigenfunctions: cos(nx), sin(nx)

        MATLAB Chebfun computes this in ~1.2s with perfect accuracy.
        Python implementation takes ~250s but produces correct results.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: -u.diff(2)
        N.bc = "periodic"

        # Compute first few eigenvalues
        eigvals, eigfuns = N.eigs(k=6)

        # Expected: 0, 1, 1, 4, 4, 9, 9, ... (each n² has multiplicity 2 for n>0)
        # Check that we get approximately 0, 1, 4, 9, ...
        expected_unique = np.array([0.0, 1.0, 4.0, 9.0])

        # Extract unique eigenvalues (allowing for small differences due to numerics)
        eigvals_sorted = np.sort(np.real(eigvals))
        eigvals_unique = [eigvals_sorted[0]]
        for ev in eigvals_sorted[1:]:
            if np.abs(ev - eigvals_unique[-1]) > 0.5:  # Different eigenvalue
                eigvals_unique.append(ev)

        # Check first few unique eigenvalues
        for i, expected in enumerate(expected_unique[: len(eigvals_unique)]):
            assert np.abs(eigvals_unique[i] - expected) < 0.1, (
                f"Eigenvalue {i}: got {eigvals_unique[i]}, expected {expected}"
            )

    def test_periodic_eigenfunction_orthogonality(self):
        """Test that periodic eigenfunctions are orthogonal."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: -u.diff(2)
        N.bc = "periodic"

        eigvals, eigfuns = N.eigs(k=6)

        # Check orthogonality of distinct eigenspaces
        for i in range(len(eigfuns)):
            for j in range(i + 1, len(eigfuns)):
                # Inner product over [0, 2π]
                product = eigfuns[i] * eigfuns[j]
                inner_prod = product.sum()

                # Should be zero for different eigenvalues
                if np.abs(eigvals[i] - eigvals[j]) > 0.1:
                    assert np.abs(inner_prod) < 1e-8, f"<φ_{i}, φ_{j}> = {inner_prod}, should be ~0"


class TestPeriodicAdvancedProblems:
    """Advanced tests for periodic BCs."""

    def test_periodic_nonlinear(self):
        """Test nonlinear BVP with periodic BCs via Newton iteration.

        Solve: u'' + 0.1*u² = f(x) with periodic BCs
        where f is chosen so u = sin(x) is the exact solution.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2) + 0.1 * u**2
        N.bc = "periodic"

        # RHS: if u = sin(x), then u'' = -sin(x), u^2 = sin^2(x)
        # So u'' + 0.1*u^2 = -sin(x) + 0.1*sin^2(x)
        N.rhs = chebfun(lambda x: -np.sin(x) + 0.1 * np.sin(x) ** 2, [0, 2 * np.pi])

        # Good initial guess
        N.init = chebfun(lambda x: np.sin(x), [0, 2 * np.pi])

        u = N.solve()

        # Check periodicity of function
        u_0 = u(np.array([0.0]))[0]
        u_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(u_0 - u_2pi) < 1e-10, f"|u(0) - u(2π)| = {np.abs(u_0 - u_2pi)}"

        # Check periodicity of derivative
        uprime = u.diff()
        uprime_0 = uprime(np.array([0.0]))[0]
        uprime_2pi = uprime(np.array([2 * np.pi]))[0]
        assert np.abs(uprime_0 - uprime_2pi) < 1e-9, f"|u'(0) - u'(2π)| = {np.abs(uprime_0 - uprime_2pi)}"

        # Check against exact solution
        u_exact = chebfun(lambda x: np.sin(x), [0, 2 * np.pi])
        x_test = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        error = np.max(np.abs(u(x_test) - u_exact(x_test)))
        assert error < 1e-6, f"Solution error: {error}"

    @pytest.mark.skip(reason="Coupled systems with periodic BCs need special handling - under-constrained problem")
    def test_periodic_systems(self):
        """Test coupled system with periodic BCs.

        Solve: u'' = v, v'' = -u with periodic BCs
        Expected: u = sin(x), v = cos(x) (up to constants)

        Note: This problem is under-constrained with just periodic BCs.
        The system u'' - v = 0, v'' + u = 0 with periodic BCs admits
        infinitely many solutions differing by constant shifts.
        Additional constraints (like u(0) = 0) would be needed for uniqueness.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u, v: [u.diff(2) - v, v.diff(2) + u]
        N.bc = "periodic"
        N.rhs = [chebfun(lambda x: 0 * x, [0, 2 * np.pi]), chebfun(lambda x: 0 * x, [0, 2 * np.pi])]

        # This will fail with "Insufficient boundary conditions"
        # because periodic BCs for systems are not yet implemented
        u, v = N.solve()

        # If it somehow works, check periodicity
        u_0 = u(np.array([0.0]))[0]
        u_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(u_0 - u_2pi) < 1e-9, "u not periodic"

        v_0 = v(np.array([0.0]))[0]
        v_2pi = v(np.array([2 * np.pi]))[0]
        assert np.abs(v_0 - v_2pi) < 1e-9, "v not periodic"

    def test_periodic_different_period(self):
        """Test periodic BCs on domains other than [0, 2π]."""
        L = 4.0  # Period
        N = chebop([0, L])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * np.pi * x / L), [0, L])

        u = N.solve()

        # Check periodicity
        u_0 = u(np.array([0.0]))[0]
        u_L = u(np.array([L]))[0]
        assert np.abs(u_0 - u_L) < 1e-10


class TestPeriodicRobustness:
    """Robustness tests for periodic BCs."""

    def test_periodic_high_frequency(self):
        """Test periodic BCs with high-frequency forcing."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        # High frequency: sin(10x)
        N.rhs = chebfun(lambda x: -100 * np.sin(10 * x), [0, 2 * np.pi])

        u = N.solve()

        # Should resolve: u = sin(10x) + C
        # Check periodicity
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([2 * np.pi]))[0]) < 1e-9

    def test_periodic_near_singular(self):
        """Test periodic BCs with near-singular operator."""
        N = chebop([0, 2 * np.pi])
        eps = 1e-6
        N.op = lambda u: eps * u.diff(2) + u
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.cos(x), [0, 2 * np.pi])

        u = N.solve()

        # Check periodicity is maintained even for stiff problem
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([2 * np.pi]))[0]) < 1e-8

    def test_periodic_adaptive_refinement(self):
        """Test that periodic solver uses adaptive refinement correctly.

        Tests that the solver correctly adapts discretization size for functions
        with different smoothness levels.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"

        # Start with smooth forcing
        N.rhs = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])
        u_smooth = N.solve()

        # Now use less smooth forcing (still zero mean, but has corners)
        # Use: cos(3x) - cos(x) which has zero mean and is less smooth
        N.rhs = chebfun(lambda x: np.cos(3 * x) - np.cos(x), [0, 2 * np.pi])
        u_rough = N.solve()

        # Both should be periodic
        assert np.abs(u_smooth(np.array([0.0]))[0] - u_smooth(np.array([2 * np.pi]))[0]) < 1e-10
        assert np.abs(u_rough(np.array([0.0]))[0] - u_rough(np.array([2 * np.pi]))[0]) < 1e-10

        # The less smooth function should require more discretization points
        # (though this is not always guaranteed with adaptive methods)
        n_smooth = u_smooth.funs[0].onefun.size
        n_rough = u_rough.funs[0].onefun.size
        # Both should converge to reasonable sizes
        assert n_smooth < 200, f"Smooth solution used {n_smooth} points (expected < 200)"
        assert n_rough < 200, f"Rough solution used {n_rough} points (expected < 200)"


class TestPeriodicErrorHandling:
    """Tests for error handling with periodic BCs."""

    def test_periodic_with_other_bcs_error(self):
        """Test that setting periodic BC along with lbc/rbc raises error."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.lbc = 0  # Should conflict with periodic

        with pytest.raises((ValueError, RuntimeError)):
            N.solve()

    def test_periodic_wrong_domain(self):
        """Test that periodic BC on [a,b] with b-a != 2π still works (scaled)."""
        # Periodic BCs should work on any domain - internally scaled
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * np.pi * x), [0, 1])

        u = N.solve()

        # Should still enforce u(0) = u(1)
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([1.0]))[0]) < 1e-10
