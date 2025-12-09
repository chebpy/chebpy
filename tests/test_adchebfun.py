"""Tests for adchebfun automatic differentiation with operator blocks.

Tests that AD correctly computes Fréchet derivatives as discrete matrices.
"""

import numpy as np
import pytest
from scipy import sparse

from chebpy import chebfun
from chebpy.adchebfun import AdChebfun, AdChebfunScalar, linearize_bc_matrix
from chebpy.spectral import diff_matrix
from chebpy.utilities import Interval


class TestAdChebfunBasic:
    """Test basic adchebfun construction and properties."""

    def test_construction_from_chebfun(self):
        """Test creating adchebfun from a chebfun."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        assert isinstance(f_ad, AdChebfun)
        assert f_ad.n == 16
        # Check domain endpoints (domain is a Domain object, not tuple)
        domain_endpoints = (f_ad.domain[0], f_ad.domain[-1])
        assert domain_endpoints == (0, 1)
        assert sparse.issparse(f_ad.jacobian)
        assert f_ad.jacobian.shape == (17, 17)

    def test_identity_jacobian(self):
        """Test that initial Jacobian is identity."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Jacobian should be identity
        J = f_ad.jacobian.toarray()
        I = np.eye(9)
        np.testing.assert_allclose(J, I, atol=1e-14)


class TestAdChebfunDifferentiation:
    """Test differentiation operator."""

    def test_diff_jacobian_is_diff_matrix(self):
        """Test that diff() produces differentiation matrix as Jacobian."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Differentiate
        f_prime_ad = f_ad.diff()

        # Jacobian should be differentiation matrix
        D = diff_matrix(8, (0, 1), order=1)
        J = f_prime_ad.jacobian

        np.testing.assert_allclose(J.toarray(), D.toarray(), atol=1e-12)

    def test_diff_twice(self):
        """Test second derivative: D @ D = D^2."""
        f = chebfun(lambda x: x**3, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Second derivative via two diff() calls
        f_prime_ad = f_ad.diff()
        f_double_prime_ad = f_prime_ad.diff()

        # Jacobian should be D^2
        D = diff_matrix(8, (0, 1), order=1)
        D2 = D @ D
        J = f_double_prime_ad.jacobian

        np.testing.assert_allclose(J.toarray(), D2.toarray(), atol=1e-10)

    def test_diff_order_2(self):
        """Test diff(order=2) directly."""
        f = chebfun(lambda x: x**3, [0, 1])
        f_ad = AdChebfun(f, n=8)

        f_double_prime_ad = f_ad.diff(order=2)

        # Should match D^2
        D2 = diff_matrix(8, (0, 1), order=2)
        np.testing.assert_allclose(f_double_prime_ad.jacobian.toarray(), D2.toarray(), atol=1e-10)


class TestAdChebfunArithmetic:
    """Test arithmetic operations."""

    def test_addition(self):
        """Test addition: (f + g)' = f' + g'."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: np.sin(x), [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        h_ad = f_ad + g_ad

        # Jacobian should be sum of Jacobians (both identity initially)
        J_f = f_ad.jacobian.toarray()
        J_g = g_ad.jacobian.toarray()
        J_h = h_ad.jacobian.toarray()

        np.testing.assert_allclose(J_h, J_f + J_g, atol=1e-14)

    def test_addition_with_constant(self):
        """Test f + c: Jacobian unchanged."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        h_ad = f_ad + 5.0

        # Jacobian should be unchanged
        np.testing.assert_allclose(h_ad.jacobian.toarray(), f_ad.jacobian.toarray(), atol=1e-14)

    def test_subtraction(self):
        """Test subtraction: (f - g)' = f' - g'."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        h_ad = f_ad - g_ad

        J_h = h_ad.jacobian.toarray()
        J_expected = f_ad.jacobian.toarray() - g_ad.jacobian.toarray()

        np.testing.assert_allclose(J_h, J_expected, atol=1e-14)

    def test_multiplication_by_constant(self):
        """Test c * f: Jacobian scaled by c."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        h_ad = 3.0 * f_ad

        np.testing.assert_allclose(h_ad.jacobian.toarray(), 3.0 * f_ad.jacobian.toarray(), atol=1e-14)

    def test_negation(self):
        """Test -f: Jacobian negated."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        h_ad = -f_ad

        np.testing.assert_allclose(h_ad.jacobian.toarray(), -f_ad.jacobian.toarray(), atol=1e-14)


class TestAdChebfunProduct:
    """Test product rule for multiplication."""

    def test_product_of_functions(self):
        """Test (f * g)' = f' * g + f * g' via Jacobian."""
        # Use simple polynomial so we can verify numerically
        def f_func(x):
            return x
        def g_func(x):
            return x**2

        f = chebfun(f_func, [0, 1])
        g = chebfun(g_func, [0, 1])

        f_ad = AdChebfun(f, n=16)
        g_ad = AdChebfun(g, n=16)

        # Compute product via AD
        h_ad = f_ad * g_ad

        # Verify function value
        h_expected = chebfun(lambda x: f_func(x) * g_func(x), [0, 1])
        x_test = np.linspace(0, 1, 10)
        np.testing.assert_allclose(h_ad.func(x_test), h_expected(x_test), atol=1e-12)

        # Verify Jacobian via finite differences
        # For small perturbation v, (f+εv)*(g+εw) ≈ f*g + ε(v*g + f*w)
        # So J_{f*g} = M_g * J_f + M_f * J_g
        # Since J_f = J_g = I initially, J_{f*g} = M_g + M_f
        from chebpy.spectral import mult_matrix
        M_f = mult_matrix(f, 16, Interval(0, 1))
        M_g = mult_matrix(g, 16, Interval(0, 1))

        J_expected = (M_g + M_f).toarray()
        J_actual = h_ad.jacobian.toarray()

        np.testing.assert_allclose(J_actual, J_expected, atol=1e-10)


class TestAdChebfunEvaluation:
    """Test evaluation operator."""

    def test_evaluation_at_point(self):
        """Test that evaluation produces correct Jacobian."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # Evaluate at x=0.5
        result = f_ad(np.array([0.5]))

        assert isinstance(result, AdChebfunScalar)
        assert result.jacobian.shape == (1, 17)

        # Value should be correct
        np.testing.assert_allclose(result.value, 0.25, atol=1e-12)

    def test_evaluation_then_subtract(self):
        """Test BC-like operation: u(a) - c."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # BC: u(1) - 1 = 0
        bc_result = f_ad(np.array([1.0])) - 1.0

        assert isinstance(bc_result, AdChebfunScalar)

        # Residual should be 0 (since x^2 at x=1 is 1)
        np.testing.assert_allclose(bc_result.value, 0.0, atol=1e-12)

        # Jacobian should be evaluation row at x=1
        assert bc_result.jacobian.shape == (1, 17)


class TestAdChebfunDerivativeBC:
    """Test derivative boundary conditions."""

    def test_derivative_bc_jacobian(self):
        """Test that u'(b) - c produces differentiation matrix row."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # BC: u'(1) - 2 = 0
        f_prime_ad = f_ad.diff()
        bc_result = f_prime_ad(np.array([1.0])) - 2.0

        # Residual should be 0 (since (x^2)' = 2x, and 2x at x=1 is 2)
        np.testing.assert_allclose(bc_result.value, 0.0, atol=1e-10)

        # Jacobian should be last row of differentiation matrix
        D = diff_matrix(16, (0, 1), order=1)
        D_last_row = D.toarray()[-1, :]

        np.testing.assert_allclose(bc_result.jacobian.toarray().ravel(), D_last_row, atol=1e-10)


class TestAdChebfunTranscendental:
    """Test transcendental functions."""

    def test_sin(self):
        """Test sin(f)' = cos(f) * f'."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=16)

        h_ad = f_ad.sin()

        # Function value
        h_expected = chebfun(lambda x: np.sin(x), [0, 1])
        x_test = np.linspace(0, 1, 10)
        np.testing.assert_allclose(h_ad.func(x_test), h_expected(x_test), atol=1e-12)

        # Jacobian should be M_cos(f) * J_f = M_cos(f) since J_f = I
        cos_f = chebfun(lambda x: np.cos(x), [0, 1])
        from chebpy.spectral import mult_matrix
        M_cos = mult_matrix(cos_f, 16, Interval(0, 1))

        np.testing.assert_allclose(h_ad.jacobian.toarray(), M_cos.toarray(), atol=1e-10)

    def test_cos(self):
        """Test cos(f)' = -sin(f) * f'."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=16)

        h_ad = f_ad.cos()

        # Jacobian should be -M_sin(f)
        neg_sin_f = chebfun(lambda x: -np.sin(x), [0, 1])
        from chebpy.spectral import mult_matrix
        M_neg_sin = mult_matrix(neg_sin_f, 16, Interval(0, 1))

        np.testing.assert_allclose(h_ad.jacobian.toarray(), M_neg_sin.toarray(), atol=1e-10)

    def test_exp(self):
        """Test exp(f)' = exp(f) * f'."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=16)

        h_ad = f_ad.exp()

        # Jacobian should be M_exp(f)
        exp_f = chebfun(lambda x: np.exp(x), [0, 1])
        from chebpy.spectral import mult_matrix
        M_exp = mult_matrix(exp_f, 16, Interval(0, 1))

        np.testing.assert_allclose(h_ad.jacobian.toarray(), M_exp.toarray(), atol=1e-10)

    def test_log(self):
        """Test log(f)' = f'/f."""
        f = chebfun(lambda x: 1 + x, [0, 1])  # Ensure positive
        f_ad = AdChebfun(f, n=16)

        h_ad = f_ad.log()

        # Jacobian should be M_{1/f}
        one_over_f = chebfun(lambda x: 1.0 / (1 + x), [0, 1])
        from chebpy.spectral import mult_matrix
        M = mult_matrix(one_over_f, 16, Interval(0, 1))

        np.testing.assert_allclose(h_ad.jacobian.toarray(), M.toarray(), atol=1e-10)

    def test_sqrt(self):
        """Test sqrt(f)' = f' / (2*sqrt(f))."""
        f = chebfun(lambda x: 1 + x, [0, 1])
        f_ad = AdChebfun(f, n=16)

        h_ad = f_ad.sqrt()

        # Jacobian should be M_{1/(2*sqrt(f))}
        coeff = chebfun(lambda x: 1.0 / (2.0 * np.sqrt(1 + x)), [0, 1])
        from chebpy.spectral import mult_matrix
        M = mult_matrix(coeff, 16, Interval(0, 1))

        np.testing.assert_allclose(h_ad.jacobian.toarray(), M.toarray(), atol=1e-10)

    def test_abs_works(self):
        """Test that abs() works with sign(u) as derivative."""
        f = chebfun(lambda x: x - 0.5, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # abs() should work now (uses sign as derivative)
        f_abs = f_ad.abs()

        # Check that the function value is correct
        testpts = np.linspace(0, 1, 10)
        expected = np.abs(f(testpts))
        actual = f_abs.func(testpts)
        assert np.allclose(actual, expected, rtol=1e-10)


class TestLinearizeBCMatrix:
    """Test linearize_bc_matrix function."""

    def test_simple_dirichlet(self):
        """Test u(a) - c linearization."""
        u = chebfun(lambda x: x**2, [0, 1])
        def bc(f):
            return f(np.array([0.0])) - 0.0

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=16)

        # Residual should be u(0) - 0 = 0
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

        # Jacobian should be evaluation row at x=0
        from chebpy.spectral import barycentric_matrix
        E = barycentric_matrix(np.array([0.0]), 16, Interval(0, 1))
        expected_row = E.toarray().ravel()

        np.testing.assert_allclose(jacobian_row, expected_row, atol=1e-12)

    def test_derivative_bc(self):
        """Test u'(b) - c linearization."""
        u = chebfun(lambda x: x**2, [0, 1])
        def bc(f):
            return f.diff()(np.array([1.0])) - 2.0

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=16)

        # Residual should be u'(1) - 2 = 2 - 2 = 0
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

        # Jacobian should be last row of D
        D = diff_matrix(16, (0, 1), order=1)
        expected_row = D.toarray()[-1, :]

        np.testing.assert_allclose(jacobian_row, expected_row, atol=1e-10)

    def test_blasius_bc_form(self):
        """Test BC of form f.diff() - 1 (Blasius-style)."""
        u = chebfun(lambda x: x * (1 - np.exp(-x)), [0, 5])
        def bc(f):
            return f.diff()(np.array([5.0])) - 1.0

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=32)

        # Jacobian should be last row of differentiation matrix
        D = diff_matrix(32, (0, 5), order=1)
        expected_row = D.toarray()[-1, :]

        # This should be EXACT, no numerical error!
        np.testing.assert_allclose(jacobian_row, expected_row, atol=1e-13)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
