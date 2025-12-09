"""Comprehensive tests for coefficient extraction from linear operators.

These tests verify that the coefficient extraction mechanism correctly
identifies the coefficient functions c_k(x) for operators of the form:

    L[u] = c_0(x)*u + c_1(x)*u' + c_2(x)*u'' + ...

The tests cover:
1. Constant coefficient operators
2. Variable coefficient operators
3. Mixed order operators
4. Higher-order operators (3rd, 4th order)
5. Edge cases (zero coefficients, very smooth coefficients)
6. Different domains ([0,1], [-1,1], [-5,5])
"""

import numpy as np
import pytest

from chebpy import chebfun, chebop


class TestConstantCoefficientExtraction:
    """Tests for operators with constant coefficients."""

    def test_identity(self):
        """L[u] = u -> c_0 = 1, c_1 = c_2 = 0."""
        N = chebop([0, 1])
        N.op = lambda u: u
        N.analyze_operator()

        coeffs = N._coeffs
        assert len(coeffs) >= 1

        x_test = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(coeffs[0](x_test), [1, 1, 1], atol=1e-10)

    def test_first_derivative(self):
        """L[u] = u' -> c_0 = 0, c_1 = 1."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff()
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        # c_0 should be 0
        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)
        # c_1 should be 1
        np.testing.assert_allclose(coeffs[1](x_test), [1, 1, 1], atol=1e-10)

    def test_second_derivative(self):
        """L[u] = u'' -> c_0 = c_1 = 0, c_2 = 1."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_laplacian_plus_constant(self):
        """L[u] = u'' + 2u -> c_0 = 2, c_1 = 0, c_2 = 1."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + 2*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [2, 2, 2], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_advection_diffusion(self):
        """L[u] = u'' + 3u' + u -> c_0 = 1, c_1 = 3, c_2 = 1."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + 3*u.diff() + u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [1, 1, 1], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [3, 3, 3], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_negative_coefficients(self):
        """L[u] = u'' - 5u' - 10u."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) - 5*u.diff() - 10*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [-10, -10, -10], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [-5, -5, -5], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)


class TestVariableCoefficientExtraction:
    """Tests for operators with variable (x-dependent) coefficients."""

    def test_xu(self):
        """L[u] = x*u -> c_0 = x."""
        N = chebop([0, 1])
        N.op = lambda x, u: x * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0.5, 1], atol=1e-10)

    def test_x_squared_u(self):
        """L[u] = x^2 * u -> c_0 = x^2."""
        N = chebop([0, 1])
        N.op = lambda x, u: x**2 * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0.25, 1], atol=1e-10)

    def test_sinx_coefficient(self):
        """L[u] = sin(πx)*u -> c_0 = sin(πx)."""
        N = chebop([0, 1])
        N.op = lambda x, u: np.sin(np.pi * x) * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])
        expected = np.sin(np.pi * x_test)

        np.testing.assert_allclose(coeffs[0](x_test), expected, atol=1e-10)

    def test_variable_second_derivative(self):
        """L[u] = u'' + x*u -> c_0 = x, c_1 = 0, c_2 = 1."""
        N = chebop([0, 1])
        N.op = lambda x, u: u.diff(2) + x * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0.5, 1], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_all_variable(self):
        """L[u] = (1+x)u'' + x*u' + (1-x)u."""
        N = chebop([0, 1])
        N.op = lambda x, u: (1+x)*u.diff(2) + x*u.diff() + (1-x)*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [1, 0.5, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0.5, 1], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1.5, 2], atol=1e-10)


class TestHigherOrderOperators:
    """Tests for 3rd and 4th order operators."""

    def test_third_order(self):
        """L[u] = u'''."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(3)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[3](x_test), [1, 1, 1], atol=1e-10)

    def test_biharmonic(self):
        """L[u] = u'''' (biharmonic)."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff(4)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        for i in range(4):
            np.testing.assert_allclose(coeffs[i](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[4](x_test), [1, 1, 1], atol=1e-10)

    def test_airy_operator(self):
        """Airy: L[u] = u'' - x*u."""
        N = chebop([-1, 1])
        N.op = lambda x, u: u.diff(2) - x*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([-1.0, 0.0, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [1, 0, -1], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)


class TestDifferentDomains:
    """Tests for operators on different domains."""

    def test_negative_domain(self):
        """L[u] = u'' on [-1, 1]."""
        N = chebop([-1, 1])
        N.op = lambda u: u.diff(2)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([-1.0, 0.0, 1.0])

        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_large_domain(self):
        """L[u] = u'' + x*u on [-5, 5]."""
        N = chebop([-5, 5])
        N.op = lambda x, u: u.diff(2) + x*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([-5.0, 0.0, 5.0])

        np.testing.assert_allclose(coeffs[0](x_test), [-5, 0, 5], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)

    def test_asymmetric_domain(self):
        """L[u] = u'' on [2, 7]."""
        N = chebop([2, 7])
        N.op = lambda u: u.diff(2)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([2.0, 4.5, 7.0])

        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-10)


class TestJacobianCoefficients:
    """Tests for coefficient extraction from Jacobians of nonlinear operators."""

    def test_u_squared_jacobian(self):
        """N(u) = u^2 at u=x -> J[u](v) = 2u*v = 2x*v
        So c_0 = 2x, c_1 = c_2 = 0.
        """
        N = chebop([0, 1])
        N.op = lambda u: u**2
        N.lbc = 0
        N.rbc = 1
        N.analyze_operator()

        u = chebfun(lambda x: x, [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 1, 2], atol=1e-9)

    def test_u_prime_squared_jacobian(self):
        """N(u) = (u')^2 at u=x (so u'=1) -> J[u](v) = 2u'*v' = 2*v'
        So c_0 = 0, c_1 = 2, c_2 = 0.
        """
        N = chebop([0, 1])
        N.op = lambda u: u.diff()**2
        N.lbc = 0
        N.rbc = 1
        N.analyze_operator()

        u = chebfun(lambda x: x, [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(coeffs[1](x_test), [2, 2, 2], atol=1e-9)

    def test_laplacian_plus_u_squared(self):
        """N(u) = u'' + u^2 at u=x -> J[u](v) = v'' + 2u*v
        So c_0 = 2x, c_1 = 0, c_2 = 1.
        """
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + u**2
        N.lbc = 0
        N.rbc = 1
        N.analyze_operator()

        u = chebfun(lambda x: x, [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 1, 2], atol=1e-9)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1, 1], atol=1e-9)

    def test_cubic_jacobian(self):
        """N(u) = u^3 at u=x -> J[u](v) = 3u^2*v = 3x^2*v
        So c_0 = 3x^2.
        """
        N = chebop([0, 1])
        N.op = lambda u: u**3
        N.lbc = 0
        N.rbc = 1
        N.analyze_operator()

        u = chebfun(lambda x: x, [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])
        expected = 3 * x_test**2

        np.testing.assert_allclose(coeffs[0](x_test), expected, atol=1e-9)

    def test_jacobian_at_nonlinear_point(self):
        """N(u) = u^2 at u=sin(πx) -> J[u](v) = 2sin(πx)*v."""
        N = chebop([0, 1])
        N.op = lambda u: u**2
        N.lbc = 0
        N.rbc = 0
        N.analyze_operator()

        u = chebfun(lambda x: np.sin(np.pi * x), [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.25, 0.5, 0.75])
        expected = 2 * np.sin(np.pi * x_test)

        np.testing.assert_allclose(coeffs[0](x_test), expected, atol=1e-9)

    def test_jacobian_u_times_uprime(self):
        """N(u) = u*u' at u=x -> J[u](v) = u'*v + u*v' = 1*v + x*v'
        So c_0 = 1, c_1 = x.
        """
        N = chebop([0, 1])
        N.op = lambda u: u * u.diff()
        N.lbc = 0
        N.rbc = 1
        N.analyze_operator()

        u = chebfun(lambda x: x, [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [1, 1, 1], atol=1e-9)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0.5, 1], atol=1e-9)

    def test_jacobian_uprime_squared(self):
        """N(u) = (u')^2 at u=sin(x) -> J[u](v) = 2u'*v' = 2cos(x)*v'."""
        N = chebop([0, 1])
        N.op = lambda u: u.diff()**2
        N.lbc = 0
        N.rbc = np.sin(1)
        N.analyze_operator()

        u = chebfun(lambda x: np.sin(x), [0, 1])
        jac = N._compute_jacobian(u)

        coeffs = jac._coeffs
        x_test = np.array([0.0, 0.5, 1.0])
        expected = 2 * np.cos(x_test)

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(coeffs[1](x_test), expected, atol=1e-9)


class TestComplexOperators:
    """Tests for more complex operator combinations."""

    def test_product_x_times_uprime(self):
        """L[u] = x*u' -> c_0 = 0, c_1 = x."""
        N = chebop([0, 1])
        N.op = lambda x, u: x * u.diff()
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0.5, 1], atol=1e-10)

    def test_product_x_squared_times_udoubleprime(self):
        """L[u] = x^2*u'' -> c_0 = c_1 = 0, c_2 = x^2."""
        N = chebop([0, 1])
        N.op = lambda x, u: x**2 * u.diff(2)
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [0, 0.25, 1], atol=1e-10)

    def test_sturm_liouville_form(self):
        """Sturm-Liouville: (p(x)u')' + q(x)u with p=1+x, q=x
        Expands to: p*u'' + p'*u' + q*u = (1+x)u'' + u' + x*u.
        """
        N = chebop([0, 1])
        # Direct form
        N.op = lambda x, u: (1+x)*u.diff(2) + u.diff() + x*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0.5, 1], atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), [1, 1, 1], atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), [1, 1.5, 2], atol=1e-10)

    def test_bessel_like(self):
        """Bessel-like: x^2*u'' + x*u' + (x^2-n^2)*u with n=1."""
        n = 1
        N = chebop([0.1, 1])  # Avoid x=0 singularity
        N.op = lambda x, u: x**2 * u.diff(2) + x * u.diff() + (x**2 - n**2) * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.1, 0.5, 1.0])

        expected_c0 = x_test**2 - n**2
        expected_c1 = x_test
        expected_c2 = x_test**2

        np.testing.assert_allclose(coeffs[0](x_test), expected_c0, atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), expected_c1, atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), expected_c2, atol=1e-10)

    def test_legendre_like(self):
        """Legendre-like: (1-x^2)*u'' - 2*x*u' + n(n+1)*u with n=2."""
        n = 2
        N = chebop([-0.9, 0.9])  # Avoid singularities at ±1
        N.op = lambda x, u: (1-x**2)*u.diff(2) - 2*x*u.diff() + n*(n+1)*u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([-0.5, 0.0, 0.5])

        expected_c0 = np.full(3, n*(n+1))  # = 6
        expected_c1 = -2 * x_test
        expected_c2 = 1 - x_test**2

        np.testing.assert_allclose(coeffs[0](x_test), expected_c0, atol=1e-10)
        np.testing.assert_allclose(coeffs[1](x_test), expected_c1, atol=1e-10)
        np.testing.assert_allclose(coeffs[2](x_test), expected_c2, atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and potential numerical issues."""

    def test_zero_operator(self):
        """L[u] = 0 (should give all zero coefficients)."""
        N = chebop([0, 1])
        N.op = lambda u: 0 * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.array([0.0, 0.5, 1.0])

        np.testing.assert_allclose(coeffs[0](x_test), [0, 0, 0], atol=1e-10)

    def test_very_smooth_coefficient(self):
        """L[u] = exp(x)*u (very smooth coefficient)."""
        N = chebop([0, 1])
        N.op = lambda x, u: np.exp(x) * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.linspace(0, 1, 10)
        expected = np.exp(x_test)

        np.testing.assert_allclose(coeffs[0](x_test), expected, atol=1e-10)

    def test_oscillatory_coefficient(self):
        """L[u] = sin(10x)*u (oscillatory coefficient)."""
        N = chebop([0, 1])
        N.op = lambda x, u: np.sin(10*x) * u
        N.analyze_operator()

        coeffs = N._coeffs
        x_test = np.linspace(0, 1, 20)
        expected = np.sin(10 * x_test)

        np.testing.assert_allclose(coeffs[0](x_test), expected, atol=1e-8)


class TestVerificationBySolving:
    """Tests that verify coefficient extraction by actually solving BVPs."""

    def test_solve_with_variable_coefficient(self):
        """Solve u'' + x*u = f and verify solution."""
        # Known solution: u = sin(πx)
        # u'' = -π²sin(πx)
        # f = -π²sin(πx) + x*sin(πx) = sin(πx)(-π² + x)

        N = chebop([0, 1])
        N.op = lambda x, u: u.diff(2) + x*u
        N.lbc = 0  # sin(0) = 0
        N.rbc = 0  # sin(π) = 0
        N.rhs = chebfun(lambda x: np.sin(np.pi*x)*(-np.pi**2 + x), [0, 1])

        u = N.solve()
        exact = chebfun(lambda x: np.sin(np.pi*x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs(u(x_test) - exact(x_test)))
        assert error < 1e-10

    def test_solve_helmholtz(self):
        """Solve u'' + k²u = 0 with exact BCs."""
        k = 3
        N = chebop([0, 1])
        N.op = lambda u: u.diff(2) + k**2 * u
        N.lbc = 1  # u(0) = 1
        N.rbc = np.cos(k)  # u(1) = cos(k)
        N.rhs = chebfun(lambda x: 0*x, [0, 1])

        u = N.solve()
        exact = chebfun(lambda x: np.cos(k*x), [0, 1])

        x_test = np.linspace(0, 1, 100)
        error = np.max(np.abs(u(x_test) - exact(x_test)))
        assert error < 1e-10
