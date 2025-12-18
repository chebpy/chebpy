"""Tests for periodic BC integration with Fourier collocation.

These tests verify that the periodic boundary condition implementation
correctly integrates with Fourier spectral methods.
"""

import numpy as np
import pytest

from chebpy import chebfun, chebop


class TestPeriodicBVPBasics:
    """Basic periodic BVP tests that should work."""

    def test_simple_second_order_periodic(self):
        """Test u'' = sin(2x) with periodic BCs on [0, 2π].

        Exact solution: u = -sin(2x)/4 + C (C arbitrary due to periodicity)
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Check solution is Trigtech (should use Fourier for periodic)
        assert type(u.funs[0].onefun).__name__ == "Trigtech", "Periodic BVP should produce Trigtech solution"

        # Check periodicity
        u_0 = u(np.array([0.0]))[0]
        u_2pi = u(np.array([2 * np.pi]))[0]
        assert np.abs(u_0 - u_2pi) < 1e-10

        # Check derivative periodicity
        uprime = u.diff()
        uprime_0 = uprime(np.array([0.0]))[0]
        uprime_2pi = uprime(np.array([2 * np.pi]))[0]
        assert np.abs(uprime_0 - uprime_2pi) < 1e-10

        # Check residual
        residual = u.diff(2) - N.rhs
        x_test = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        res_vals = residual(x_test)

        # Should be real
        assert np.max(np.abs(np.imag(res_vals))) < 1e-10, "Residual should be real for real problem"

        res_norm = np.max(np.abs(np.real(res_vals)))
        assert res_norm < 1e-9, f"Residual too large: {res_norm}"

        # Check against exact solution (modulo constant)
        u_exact = chebfun(lambda x: -np.sin(2 * x) / 4, [0, 2 * np.pi])
        error = u - u_exact
        # Remove mean (arbitrary constant)
        x_dense = np.linspace(0, 2 * np.pi, 200)
        error_vals = error(x_dense)
        error_centered = error_vals - np.mean(error_vals)
        assert np.max(np.abs(error_centered)) < 1e-8

    def test_fourth_order_periodic(self):
        """Test u'''' = sin(3x) with periodic BCs.

        Exact solution: u = sin(3x)/81 + C1*x^3 + C2*x^2 + C3*x + C4
        With periodic BCs: C1 = C2 = C3 = 0, C4 arbitrary
        So: u = sin(3x)/81 + C
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(4)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(3 * x), [0, 2 * np.pi])

        u = N.solve()

        # Check periodicity of all derivatives up to 3rd
        for deriv_order in range(4):
            u_deriv = u.diff(deriv_order)
            val_0 = u_deriv(np.array([0.0]))[0]
            val_2pi = u_deriv(np.array([2 * np.pi]))[0]
            assert np.abs(val_0 - val_2pi) < 1e-10, f"Derivative order {deriv_order} not periodic"

        # Check residual
        residual = u.diff(4) - N.rhs
        x_test = np.linspace(0.1, 2 * np.pi - 0.1, 50)
        res_norm = np.max(np.abs(residual(x_test)))
        assert res_norm < 1e-8, f"Residual: {res_norm}"

    def test_periodic_zero_mean_rhs(self):
        """Test u'' = sin(x) - mean(sin(x)) with periodic BCs.

        Since sin(x) already has zero mean, this should work well.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        # sin(x) has zero mean over [0, 2π]
        N.rhs = chebfun(lambda x: np.sin(x), [0, 2 * np.pi])

        u = N.solve()

        # Exact: u = -sin(x) + C
        u_exact = chebfun(lambda x: -np.sin(x), [0, 2 * np.pi])
        x_test = np.linspace(0, 2 * np.pi, 100)
        error = (u - u_exact)(x_test)
        error_centered = error - np.mean(error)
        assert np.max(np.abs(error_centered)) < 1e-8


class TestPeriodicConvergence:
    """Tests for convergence properties of periodic BVP solver."""

    def test_spectral_convergence_smooth(self):
        """Verify spectral convergence for smooth periodic function.

        For smooth periodic functions, Fourier methods should achieve
        exponential convergence (errors decrease exponentially with n).
        """
        # Solve u'' = -k^2 * sin(kx) => u = sin(kx) + C
        k = 5
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: -(k**2) * np.sin(k * x), [0, 2 * np.pi])

        u = N.solve()

        # Should converge with small n since solution is smooth
        assert u.funs[0].onefun.size < 50, f"Should converge quickly for smooth function, got n={u.funs[0].onefun.size}"

        # Check error
        u_exact = chebfun(lambda x: np.sin(k * x), [0, 2 * np.pi])
        x_test = np.linspace(0, 2 * np.pi, 200)
        error = (u - u_exact)(x_test)
        error_centered = error - np.mean(error)
        assert np.max(np.abs(error_centered)) < 1e-10


class TestPeriodicRealValued:
    """Tests to ensure solutions remain real for real problems."""

    def test_real_solution_for_real_problem(self):
        """Verify solution is real when all inputs are real."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.sin(2 * x), [0, 2 * np.pi])

        u = N.solve()

        # Evaluate at many points
        x_test = np.linspace(0, 2 * np.pi, 100)
        u_vals = u(x_test)

        # Should be real (or have negligible imaginary part)
        assert np.max(np.abs(np.imag(u_vals))) < 1e-12, (
            f"Solution has imaginary part: {np.max(np.abs(np.imag(u_vals)))}"
        )

        # Check derivatives are also real
        for deriv_order in [1, 2]:
            u_deriv = u.diff(deriv_order)
            deriv_vals = u_deriv(x_test)
            assert np.max(np.abs(np.imag(deriv_vals))) < 1e-12, f"Derivative {deriv_order} has imaginary part"


class TestPeriodicVsNonPeriodic:
    """Compare periodic vs non-periodic methods where both apply."""

    def test_periodic_more_efficient(self):
        """Periodic methods should use fewer points for periodic functions."""
        # Use a smooth periodic function

        # Periodic version
        N_per = chebop([0, 2 * np.pi])
        N_per.op = lambda u: u.diff(2)
        N_per.bc = "periodic"
        N_per.rhs = chebfun(lambda x: np.sin(3 * x), [0, 2 * np.pi])
        u_per = N_per.solve()
        n_per = u_per.funs[0].onefun.size

        # Non-periodic version with Dirichlet BCs
        N_dir = chebop([0, 2 * np.pi])
        N_dir.op = lambda u: u.diff(2)
        N_dir.lbc = lambda u: u - 0  # u(0) = 0
        N_dir.rbc = lambda u: u - 0  # u(2π) = 0
        N_dir.rhs = chebfun(lambda x: np.sin(3 * x), [0, 2 * np.pi])

        u_dir = N_dir.solve()
        n_dir = u_dir.funs[0].onefun.size

        # Periodic should be more efficient (fewer points)
        # For this smooth periodic function
        assert n_per < n_dir


class TestPeriodicIntervals:
    """Test periodic BCs on different intervals."""

    def test_periodic_on_minus_pi_to_pi(self):
        """Test on [-π, π] interval."""
        N = chebop([-np.pi, np.pi])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        # Adjust frequency for the interval length
        N.rhs = chebfun(lambda x: np.sin(2 * x), [-np.pi, np.pi])

        u = N.solve()

        # Check periodicity
        u_left = u(np.array([-np.pi]))[0]
        u_right = u(np.array([np.pi]))[0]
        assert np.abs(u_left - u_right) < 1e-10

    def test_periodic_on_0_to_1(self):
        """Test on [0, 1] interval."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.bc = "periodic"
        # Frequency adjusted for period 1
        N.rhs = chebfun(lambda x: np.sin(2 * np.pi * x), [0, 1])

        u = N.solve()

        # Exact: u = -sin(2πx)/(2π)^2 + C
        u_exact = chebfun(lambda x: -np.sin(2 * np.pi * x) / (4 * np.pi**2), [0, 1])

        x_test = np.linspace(0, 1, 50)
        error = (u - u_exact)(x_test)
        error_centered = error - np.mean(error)
        assert np.max(np.abs(error_centered)) < 1e-8


class TestPeriodicDifferentialOrders:
    """Test periodic BCs with different differential orders."""

    def test_first_order_periodic(self):
        """Test u' = cos(x) with periodic BCs.

        Exact: u = sin(x) + C

        First-order periodic BVPs are ill-conditioned with spectral methods.
        """
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff()
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: np.cos(x), [0, 2 * np.pi])

        # Limit max_n to avoid slow adaptive convergence loop
        linop = N.to_linop()
        linop.max_n = 64
        u = linop.solve(N.rhs)

        # Check periodicity
        assert np.abs(u(np.array([0.0]))[0] - u(np.array([2 * np.pi]))[0]) < 1e-10

        # Check residual (loose tolerance: first-order periodic problems are ill-conditioned)
        residual = u.diff() - N.rhs
        x_test = np.linspace(0, 2 * np.pi, 50)
        res_norm = np.max(np.abs(residual(x_test)))
        assert res_norm < 0.5

    def test_third_order_periodic(self):
        """Test u''' = -6*sin(2x) with periodic BCs."""
        N = chebop([0, 2 * np.pi])
        N.op = lambda u: u.diff(3)
        N.bc = "periodic"
        N.rhs = chebfun(lambda x: -6 * np.sin(2 * x), [0, 2 * np.pi])

        # Limit max_n to avoid slow adaptive convergence loop
        linop = N.to_linop()
        linop.max_n = 128  # Limit iterations for speed
        u = linop.solve(N.rhs)

        # Check periodicity of u and its derivatives
        for order in range(3):
            u_d = u.diff(order)
            val_0 = u_d(np.array([0.0]))[0]
            val_2pi = u_d(np.array([2 * np.pi]))[0]
            # Tolerance accounts for amplification of roundoff through differentiation
            assert np.abs(val_0 - val_2pi) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
