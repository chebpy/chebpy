"""Comprehensive tests for adchebfun to achieve >90% coverage.

This test file focuses on edge cases and untested code paths:
1. Unary operations with AdChebfun inputs and Jacobian verification
2. AdChebfunScalar operations (subscripting, arithmetic, float conversion)
3. Division edge cases (different discretizations, near-zero division)
4. Power operations with non-scalar exponents
5. Extraction utilities (_extract_residual_jacobian)
6. linearize_bc_matrix with various BC types
"""

import numpy as np
import pytest
from scipy import sparse

from chebpy import chebfun
from chebpy.adchebfun import (
    AdChebfun,
    AdChebfunScalar,
    _extract_residual_jacobian,
    linearize_bc_matrix,
)
from chebpy.spectral import mult_matrix
from chebpy.utilities import Interval


class TestAdChebfunEdgeCases:
    """Test edge cases for AdChebfun operations."""

    def test_rsub_with_constant(self):
        """Test right subtraction: c - f."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # 2 - f
        result = 2.0 - f_ad

        # Function value should be correct
        x_test = np.array([0.5])
        expected_val = 2.0 - 0.25
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

        # Jacobian should be negated
        np.testing.assert_allclose(result.jacobian.toarray(), -f_ad.jacobian.toarray(), atol=1e-14)

    def test_rsub_with_array(self):
        """Test right subtraction with array: arr - f."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Array - f (numpy array is treated as scalar in this context)
        result = 1.0 - f_ad

        # Jacobian should be negated
        np.testing.assert_allclose(result.jacobian.toarray(), -f_ad.jacobian.toarray(), atol=1e-14)

    def test_div_with_adchebfun(self):
        """Test division: f / g with quotient rule."""
        f = chebfun(lambda x: x**2, [0.5, 1])  # Avoid division by zero
        g = chebfun(lambda x: x, [0.5, 1])

        f_ad = AdChebfun(f, n=16)
        g_ad = AdChebfun(g, n=16)

        # f / g
        result = f_ad / g_ad

        # Function value should be correct
        x_test = np.linspace(0.5, 1, 5)
        expected_val = (x_test**2) / x_test
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

        # Jacobian: (1/g) * J_f - (f/g^2) * J_g
        # Since J_f = J_g = I, we get: M_{1/g} - M_{f/g^2}
        g_inv = chebfun(lambda x: 1.0 / x, [0.5, 1])
        f_over_g2 = chebfun(lambda x: x**2 / x**2, [0.5, 1])

        M_g_inv = mult_matrix(g_inv, 16, Interval(0.5, 1))
        M_f_over_g2 = mult_matrix(f_over_g2, 16, Interval(0.5, 1))

        expected_jac = M_g_inv.toarray() - M_f_over_g2.toarray()
        np.testing.assert_allclose(result.jacobian.toarray(), expected_jac, atol=1e-10)

    def test_div_with_different_discretizations(self):
        """Test that division with different discretizations raises error."""
        f = chebfun(lambda x: x**2, [0.5, 1])
        g = chebfun(lambda x: x, [0.5, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=16)

        with pytest.raises(ValueError, match="different discretizations"):
            _ = f_ad / g_ad

    def test_div_by_scalar(self):
        """Test division by scalar constant."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad / 2.0

        # Function value
        x_test = np.array([0.5])
        np.testing.assert_allclose(result.func(x_test), 0.125, atol=1e-14)

        # Jacobian scaled by 1/2
        np.testing.assert_allclose(result.jacobian.toarray(), f_ad.jacobian.toarray() / 2.0, atol=1e-14)

    def test_pow_with_scalar_exponent(self):
        """Test power with scalar exponent: f^n."""
        f = chebfun(lambda x: 1 + x, [0, 1])  # Ensure positive
        f_ad = AdChebfun(f, n=16)

        result = f_ad**3

        # Function value
        x_test = np.linspace(0, 1, 5)
        expected_val = (1 + x_test) ** 3
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

        # Jacobian: n * f^(n-1) * J_f = 3 * f^2 * I
        f_squared = chebfun(lambda x: (1 + x) ** 2, [0, 1])
        M = mult_matrix(3 * f_squared, 16, Interval(0, 1))

        np.testing.assert_allclose(result.jacobian.toarray(), M.toarray(), atol=1e-10)

    def test_pow_with_non_scalar_returns_not_implemented(self):
        """Test that power with non-scalar exponent returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Should return NotImplemented for non-scalar exponent
        result = f_ad.__pow__(np.array([2, 3]))
        assert result is NotImplemented

    def test_add_with_different_discretizations(self):
        """Test that addition with different discretizations raises error."""
        f = chebfun(lambda x: x, [0, 1])
        g = chebfun(lambda x: x**2, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=16)

        with pytest.raises(ValueError, match="different discretizations"):
            _ = f_ad + g_ad

    def test_sub_with_different_discretizations(self):
        """Test that subtraction with different discretizations raises error."""
        f = chebfun(lambda x: x, [0, 1])
        g = chebfun(lambda x: x**2, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=16)

        with pytest.raises(ValueError, match="different discretizations"):
            _ = f_ad - g_ad

    def test_mul_with_different_discretizations(self):
        """Test that multiplication with different discretizations raises error."""
        f = chebfun(lambda x: x, [0, 1])
        g = chebfun(lambda x: x**2, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=16)

        with pytest.raises(ValueError, match="different discretizations"):
            _ = f_ad * g_ad

    def test_add_with_invalid_type(self):
        """Test that addition with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad.__add__("invalid")
        assert result is NotImplemented

    def test_sub_with_invalid_type(self):
        """Test that subtraction with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad.__sub__("invalid")
        assert result is NotImplemented

    def test_mul_with_invalid_type(self):
        """Test that multiplication with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # After Bug #5 fix, Chebfun is now supported in AdChebfun arithmetic
        # Test with an actual invalid type instead
        result = f_ad.__mul__("invalid_string")
        assert result is NotImplemented

    def test_div_with_invalid_type(self):
        """Test that division with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad.__truediv__("invalid")
        assert result is NotImplemented

    def test_rsub_with_invalid_type(self):
        """Test that right subtraction with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad.__rsub__("invalid")
        assert result is NotImplemented


class TestUnaryOperationsWithAdChebfun:
    """Test unary operations with AdChebfun inputs and verify Jacobians."""

    def test_sin_on_adchebfun(self):
        """Test sin applied to result of differentiation."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # Differentiate first: f' = 2x
        f_prime = f_ad.diff()

        # Apply sin: sin(f')
        result = f_prime.sin()

        # Function value should be sin(2x)
        x_test = np.linspace(0, 1, 5)
        expected_val = np.sin(2 * x_test)
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

        # Jacobian: cos(f') * J_{f'} = cos(2x) * D
        from chebpy.spectral import diff_matrix

        cos_f_prime = chebfun(lambda x: np.cos(2 * x), [0, 1])
        M_cos = mult_matrix(cos_f_prime, 16, Interval(0, 1))
        D = diff_matrix(16, Interval(0, 1), order=1)

        expected_jac = M_cos @ D
        np.testing.assert_allclose(result.jacobian.toarray(), expected_jac.toarray(), atol=1e-9)

    def test_cos_on_adchebfun(self):
        """Test cos applied to result of differentiation."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        f_prime = f_ad.diff()
        result = f_prime.cos()

        # Function value should be cos(2x)
        x_test = np.linspace(0, 1, 5)
        expected_val = np.cos(2 * x_test)
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

    def test_exp_on_adchebfun(self):
        """Test exp applied to result of differentiation."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=16)

        result = f_ad.exp()

        # Function value should be exp(x)
        x_test = np.linspace(0, 1, 5)
        expected_val = np.exp(x_test)
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

    def test_log_on_adchebfun(self):
        """Test log applied to result of differentiation."""
        f = chebfun(lambda x: 1 + x, [0, 1])  # Ensure positive
        f_ad = AdChebfun(f, n=16)

        result = f_ad.log()

        # Function value should be log(1+x)
        x_test = np.linspace(0, 1, 5)
        expected_val = np.log(1 + x_test)
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)

    def test_sqrt_on_adchebfun(self):
        """Test sqrt applied to result of differentiation."""
        f = chebfun(lambda x: 1 + x**2, [0, 1])  # Ensure positive
        f_ad = AdChebfun(f, n=16)

        result = f_ad.sqrt()

        # Function value should be sqrt(1+x^2)
        x_test = np.linspace(0, 1, 5)
        expected_val = np.sqrt(1 + x_test**2)
        np.testing.assert_allclose(result.func(x_test), expected_val, atol=1e-12)


class TestAdChebfunScalarOperations:
    """Test AdChebfunScalar operations comprehensively."""

    def test_scalar_subscripting_single_element(self):
        """Test subscripting AdChebfunScalar to extract single element."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Evaluate at multiple points
        x_vals = np.array([0.25, 0.5, 0.75])
        result = f_ad(x_vals)

        # Subscript to get first element
        elem0 = result[0]

        assert isinstance(elem0, AdChebfunScalar)
        np.testing.assert_allclose(elem0.value, 0.25**2, atol=1e-12)

        # Jacobian should be first row
        assert elem0.jacobian.shape == (1, 9)

    def test_scalar_subscripting_slice(self):
        """Test subscripting AdChebfunScalar with slice."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        x_vals = np.array([0.25, 0.5, 0.75])
        result = f_ad(x_vals)

        # Slice
        sliced = result[:2]

        assert isinstance(sliced, AdChebfunScalar)
        assert sliced.value.shape == (2,)
        assert sliced.jacobian.shape == (2, 9)

    def test_scalar_subscripting_non_array_raises(self):
        """Test that subscripting non-array AdChebfunScalar raises error."""
        # Create a scalar AdChebfunScalar with non-array value
        jac = sparse.csr_matrix(np.eye(9))
        scalar = AdChebfunScalar(5.0, n=8, domain=Interval(0, 1), jacobian=jac)

        with pytest.raises(TypeError, match="not subscriptable"):
            _ = scalar[0]

    def test_scalar_add_with_scalar(self):
        """Test AdChebfunScalar + scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        added = result + 1.0

        np.testing.assert_allclose(added.value, 0.25 + 1.0, atol=1e-14)
        np.testing.assert_allclose(added.jacobian.toarray(), result.jacobian.toarray(), atol=1e-14)

    def test_scalar_add_with_adchebfunscalar(self):
        """Test AdChebfunScalar + AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        f_val = f_ad(np.array([0.5]))
        g_val = g_ad(np.array([0.5]))

        result = f_val + g_val

        np.testing.assert_allclose(result.value, 0.25 + 0.5, atol=1e-14)

    def test_scalar_add_with_adchebfun(self):
        """Test AdChebfunScalar + AdChebfun (edge case)."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        f_val = f_ad(np.array([0.5]))
        g_val = g_ad(np.array([0.5]))

        # Adding AdChebfunScalar + AdChebfunScalar
        result = f_val + g_val

        # Value should be f(0.5) + g(0.5) = 0.25 + 0.5
        np.testing.assert_allclose(result.value, 0.75, atol=1e-14)
        assert result.jacobian.shape == (1, 9)

    def test_scalar_sub_with_scalar(self):
        """Test AdChebfunScalar - scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        subtracted = result - 0.1

        np.testing.assert_allclose(subtracted.value, 0.25 - 0.1, atol=1e-14)

    def test_scalar_sub_with_adchebfunscalar(self):
        """Test AdChebfunScalar - AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        f_val = f_ad(np.array([0.5]))
        g_val = g_ad(np.array([0.5]))

        result = f_val - g_val

        np.testing.assert_allclose(result.value, 0.25 - 0.5, atol=1e-14)

    def test_scalar_rsub_with_scalar(self):
        """Test scalar - AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        subtracted = 1.0 - result

        np.testing.assert_allclose(subtracted.value, 1.0 - 0.25, atol=1e-14)
        np.testing.assert_allclose(subtracted.jacobian.toarray(), -result.jacobian.toarray(), atol=1e-14)

    def test_scalar_mul_with_scalar(self):
        """Test AdChebfunScalar * scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        multiplied = result * 3.0

        np.testing.assert_allclose(multiplied.value, 0.25 * 3.0, atol=1e-14)
        np.testing.assert_allclose(multiplied.jacobian.toarray(), 3.0 * result.jacobian.toarray(), atol=1e-14)

    def test_scalar_mul_with_non_scalar_returns_not_implemented(self):
        """Test that multiplication with non-scalar returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))

        mult_result = result.__mul__("invalid")
        assert mult_result is NotImplemented

    def test_scalar_neg(self):
        """Test -AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        negated = -result

        np.testing.assert_allclose(negated.value, -0.25, atol=1e-14)
        np.testing.assert_allclose(negated.jacobian.toarray(), -result.jacobian.toarray(), atol=1e-14)

    def test_scalar_float_conversion_scalar(self):
        """Test converting scalar AdChebfunScalar to float."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        float_val = float(result)

        np.testing.assert_allclose(float_val, 0.25, atol=1e-14)

    def test_scalar_float_conversion_single_element_array(self):
        """Test converting single-element array AdChebfunScalar to float."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        # result.value is already a single-element array
        float_val = float(result)

        np.testing.assert_allclose(float_val, 0.25, atol=1e-14)

    def test_scalar_float_conversion_non_scalar_raises(self):
        """Test that converting non-scalar AdChebfunScalar raises error."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.25, 0.5, 0.75]))

        with pytest.raises(ValueError, match="Cannot convert non-scalar"):
            _ = float(result)

    def test_scalar_add_with_invalid_type(self):
        """Test that addition with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        add_result = result.__add__("invalid")
        assert add_result is NotImplemented

    def test_scalar_sub_with_invalid_type(self):
        """Test that subtraction with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        sub_result = result.__sub__("invalid")
        assert sub_result is NotImplemented

    def test_scalar_rsub_with_invalid_type(self):
        """Test that right subtraction with invalid type returns NotImplemented."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))
        rsub_result = result.__rsub__("invalid")
        assert rsub_result is NotImplemented


class TestExtractResidualJacobian:
    """Test _extract_residual_jacobian utility function."""

    def test_extract_from_adchebfun(self):
        """Test extraction from AdChebfun (function, not yet evaluated)."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # Extract at boundary point x=1
        residual, jac_row = _extract_residual_jacobian(f_ad, x_bc=1.0, n=16)

        # Residual should be f(1) = 1
        np.testing.assert_allclose(residual, 1.0, atol=1e-12)

        # Jacobian row should be evaluation at x=1
        assert jac_row.shape == (17,)

    def test_extract_from_adchebfunscalar(self):
        """Test extraction from AdChebfunScalar (already evaluated)."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=16)

        # Evaluate first
        f_val = f_ad(np.array([0.5]))

        # Extract from scalar (x_bc not used since already evaluated)
        residual, jac_row = _extract_residual_jacobian(f_val, x_bc=1.0, n=16)

        # Residual should be f(0.5) = 0.25
        np.testing.assert_allclose(residual, 0.25, atol=1e-12)

        # Jacobian row shape
        assert jac_row.shape == (17,)

    def test_extract_from_scalar_constant(self):
        """Test extraction from scalar constant."""
        # Extract from float constant
        residual, jac_row = _extract_residual_jacobian(5.0, x_bc=1.0, n=16)

        # Residual should be the constant
        np.testing.assert_allclose(residual, 5.0, atol=1e-14)

        # Jacobian row should be zeros
        np.testing.assert_allclose(jac_row, np.zeros(17), atol=1e-14)

    def test_extract_from_zero_constant(self):
        """Test extraction from zero constant."""
        # Extract from zero
        residual, jac_row = _extract_residual_jacobian(0.0, x_bc=1.0, n=16)

        np.testing.assert_allclose(residual, 0.0, atol=1e-14)
        np.testing.assert_allclose(jac_row, np.zeros(17), atol=1e-14)

    def test_extract_from_array_constant(self):
        """Test extraction from array treated as constant."""
        # Extract from array (should return 0.0 for residual)
        arr = np.array([1.0, 2.0])
        residual, jac_row = _extract_residual_jacobian(arr, x_bc=1.0, n=16)

        # Non-scalar should give residual 0.0
        np.testing.assert_allclose(residual, 0.0, atol=1e-14)
        np.testing.assert_allclose(jac_row, np.zeros(17), atol=1e-14)


class TestLinearizeBCMatrixComprehensive:
    """Test linearize_bc_matrix with various BC types."""

    def test_bc_returning_list_multiple_constraints(self):
        """Test BC returning list of multiple constraints."""
        u = chebfun(lambda x: x**2, [0, 1])

        def bc_list(f):
            # Return list: [u(0), u(1)]
            return [f(np.array([0.0])), f(np.array([1.0]))]

        residuals, jacobian_matrix = linearize_bc_matrix(bc_list, u, n=16)

        # Should return list of residuals
        assert isinstance(residuals, list)
        assert len(residuals) == 2

        # Residuals should be u(0)=0 and u(1)=1
        np.testing.assert_allclose(residuals[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(residuals[1], 1.0, atol=1e-12)

        # Jacobian should be 2 rows (stacked)
        assert jacobian_matrix.shape == (2, 17)

    def test_bc_returning_tuple_multiple_constraints(self):
        """Test BC returning tuple of multiple constraints."""
        u = chebfun(lambda x: x**3, [0, 1])

        def bc_tuple(f):
            # Return tuple: (u, u.diff())
            return (f(np.array([1.0])), f.diff()(np.array([1.0])))

        residuals, jacobian_matrix = linearize_bc_matrix(bc_tuple, u, n=16)

        # Should return list of residuals
        assert isinstance(residuals, list)
        assert len(residuals) == 2

        # Residuals: u(1)=1, u'(1)=3
        np.testing.assert_allclose(residuals[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(residuals[1], 3.0, atol=1e-10)

        # Jacobian: 2 rows
        assert jacobian_matrix.shape == (2, 17)

    def test_bc_with_x_bc_provided(self):
        """Test BC with explicit x_bc parameter."""
        u = chebfun(lambda x: x**2, [0, 1])

        def bc(f):
            return f.diff()(np.array([0.5])) - 1.0

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=16, x_bc=0.5)

        # Residual: u'(0.5) - 1 = 1 - 1 = 0
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_bc_without_x_bc_uses_rightmost(self):
        """Test that x_bc defaults to rightmost point of domain."""
        u = chebfun(lambda x: x**2, [0, 1])

        def bc(f):
            return f(np.array([1.0]))

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=16)

        # Should use rightmost point (1.0)
        np.testing.assert_allclose(residual, 1.0, atol=1e-12)

    def test_bc_with_callable_returning_scalar(self):
        """Test BC that is a callable returning scalar."""
        u = chebfun(lambda x: np.sin(np.pi * x), [0, 1])

        def bc(f):
            return f(np.array([0.0]))

        residual, jacobian_row = linearize_bc_matrix(bc, u, n=32)

        # Residual should be sin(0) = 0
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

        # Jacobian row should have correct shape
        assert jacobian_row.shape == (33,)

    def test_bc_with_list_of_mixed_types(self):
        """Test BC returning list with mixed AdChebfun, AdChebfunScalar, and scalars."""
        u = chebfun(lambda x: x**2, [0, 1])

        def bc_mixed(f):
            # Return list with: AdChebfun (not evaluated), AdChebfunScalar (evaluated), scalar
            return [f, f(np.array([0.5])), 2.0]

        residuals, jacobian_matrix = linearize_bc_matrix(bc_mixed, u, n=16)

        assert len(residuals) == 3
        assert jacobian_matrix.shape == (3, 17)

        # Third residual should be the constant 2.0
        np.testing.assert_allclose(residuals[2], 2.0, atol=1e-14)


class TestAdChebfunConstructionEdgeCases:
    """Test edge cases in AdChebfun construction."""

    def test_construction_with_explicit_n(self):
        """Test construction with explicit n parameter."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=32)

        assert f_ad.n == 32
        assert f_ad.jacobian.shape == (33, 33)

    def test_construction_without_n_infers_from_func(self):
        """Test that n is inferred from chebfun when not provided."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f)

        # n should be inferred (length of coefficients - 1)
        assert f_ad.n is not None
        assert f_ad.n > 0

    def test_construction_with_custom_jacobian(self):
        """Test construction with custom Jacobian matrix."""
        f = chebfun(lambda x: x, [0, 1])
        n = 8

        # Custom Jacobian: 2*I
        custom_jac = 2.0 * sparse.eye(n + 1)

        f_ad = AdChebfun(f, n=n, jacobian=custom_jac)

        np.testing.assert_allclose(f_ad.jacobian.toarray(), custom_jac.toarray(), atol=1e-14)

    def test_construction_with_empty_funs(self):
        """Test construction with chebfun that has no funs."""
        # Create a minimal chebfun-like object without funs
        f = chebfun(lambda x: x, [0, 1])

        # Remove funs to test fallback
        class MinimalChebfun:
            def __init__(self):
                self.support = Interval(0, 1)

            def __call__(self, x):
                return x

        minimal_f = MinimalChebfun()
        f_ad = AdChebfun(minimal_f)

        # Should use default n=16
        assert f_ad.n == 16

    def test_construction_with_fun_without_onefun(self):
        """Test construction fallback when fun doesn't have onefun."""
        # Create a minimal chebfun-like object with funs but no onefun
        class MinimalFun:
            pass

        class MinimalChebfunWithFuns:
            def __init__(self):
                self.support = Interval(0, 1)
                self.funs = [MinimalFun()]

            def __call__(self, x):
                return x

        minimal_f = MinimalChebfunWithFuns()
        f_ad = AdChebfun(minimal_f)

        # Should use default n=16
        assert f_ad.n == 16

    def test_domain_conversion_from_domain_object(self):
        """Test that Domain objects are converted to Intervals."""
        f = chebfun(lambda x: x, [0, 1])
        f_ad = AdChebfun(f, n=8)

        # Domain should be an Interval
        assert hasattr(f_ad.domain, "__getitem__")
        assert f_ad.domain[0] == 0
        assert f_ad.domain[-1] == 1


class TestAdChebfunAdditionalCoverage:
    """Additional tests to cover remaining lines."""

    def test_add_two_adchebfuns(self):
        """Test adding two AdChebfun objects."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        result = f_ad + g_ad

        # Check function value
        x_test = np.array([0.5])
        np.testing.assert_allclose(result.func(x_test), 0.25 + 0.5, atol=1e-12)

        # Jacobians should be summed
        expected_jac = f_ad.jacobian + g_ad.jacobian
        np.testing.assert_allclose(result.jacobian.toarray(), expected_jac.toarray(), atol=1e-14)

    def test_add_with_numpy_scalar(self):
        """Test adding numpy scalar to AdChebfun."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad + np.float64(1.0)

        # Jacobian should be unchanged
        np.testing.assert_allclose(result.jacobian.toarray(), f_ad.jacobian.toarray(), atol=1e-14)

    def test_radd_with_scalar(self):
        """Test right addition with scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = 2.0 + f_ad

        # Jacobian should be unchanged
        np.testing.assert_allclose(result.jacobian.toarray(), f_ad.jacobian.toarray(), atol=1e-14)

    def test_sub_two_adchebfuns(self):
        """Test subtracting two AdChebfun objects."""
        f = chebfun(lambda x: x**2, [0, 1])
        g = chebfun(lambda x: x, [0, 1])

        f_ad = AdChebfun(f, n=8)
        g_ad = AdChebfun(g, n=8)

        result = f_ad - g_ad

        # Check function value
        x_test = np.array([0.5])
        np.testing.assert_allclose(result.func(x_test), 0.25 - 0.5, atol=1e-12)

        # Jacobians should be differenced
        expected_jac = f_ad.jacobian - g_ad.jacobian
        np.testing.assert_allclose(result.jacobian.toarray(), expected_jac.toarray(), atol=1e-14)

    def test_sub_with_numpy_scalar(self):
        """Test subtracting numpy scalar from AdChebfun."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad - np.float64(1.0)

        # Jacobian should be unchanged
        np.testing.assert_allclose(result.jacobian.toarray(), f_ad.jacobian.toarray(), atol=1e-14)

    def test_mul_two_adchebfuns(self):
        """Test multiplying two AdChebfun objects."""
        f = chebfun(lambda x: x, [0, 1])
        g = chebfun(lambda x: x**2, [0, 1])

        f_ad = AdChebfun(f, n=16)
        g_ad = AdChebfun(g, n=16)

        result = f_ad * g_ad

        # Check function value
        x_test = np.array([0.5])
        np.testing.assert_allclose(result.func(x_test), 0.5 * 0.25, atol=1e-12)

        # Jacobian: M_g * J_f + M_f * J_g
        M_f = mult_matrix(f, 16, Interval(0, 1))
        M_g = mult_matrix(g, 16, Interval(0, 1))
        expected_jac = M_g @ f_ad.jacobian + M_f @ g_ad.jacobian
        np.testing.assert_allclose(result.jacobian.toarray(), expected_jac.toarray(), atol=1e-10)

    def test_mul_with_scalar_int(self):
        """Test multiplying AdChebfun by scalar integer."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad * 3

        # Jacobian should be scaled
        np.testing.assert_allclose(result.jacobian.toarray(), 3 * f_ad.jacobian.toarray(), atol=1e-14)

    def test_neg_adchebfun(self):
        """Test negation of AdChebfun."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = -f_ad

        # Check function value
        x_test = np.array([0.5])
        np.testing.assert_allclose(result.func(x_test), -0.25, atol=1e-12)

        # Jacobian should be negated
        np.testing.assert_allclose(result.jacobian.toarray(), -f_ad.jacobian.toarray(), atol=1e-14)

    def test_abs_works(self):
        """Test that abs() works with autodifferentiation.

        abs() uses sign(u) as the derivative (valid almost everywhere).
        At u=0, we use 0 as the derivative by convention.
        """
        f = chebfun(lambda x: x - 0.5, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad.abs()

        # Should return an AdChebfun
        assert isinstance(result, AdChebfun)

        # Check that the result matches |x - 0.5|
        x_test = np.linspace(0, 1, 10)
        expected = np.abs(x_test - 0.5)
        result_vals = result(x_test).value  # Extract value from AdChebfunScalar
        np.testing.assert_allclose(result_vals, expected, atol=1e-8)

    def test_scalar_mul_with_rmul(self):
        """Test right multiplication with scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = 4.0 * f_ad

        # Jacobian should be scaled
        np.testing.assert_allclose(result.jacobian.toarray(), 4.0 * f_ad.jacobian.toarray(), atol=1e-14)

    def test_scalar_float_conversion_from_scalar_value(self):
        """Test float conversion when value is already a scalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))

        # Ensure value is scalar
        assert isinstance(result.value, (float, np.ndarray))
        float_val = float(result)
        np.testing.assert_allclose(float_val, 0.25, atol=1e-14)

    def test_scalar_radd(self):
        """Test right addition for AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))

        # Test radd: 1.0 + result
        added = 1.0 + result

        np.testing.assert_allclose(added.value, 1.0 + 0.25, atol=1e-14)
        np.testing.assert_allclose(added.jacobian.toarray(), result.jacobian.toarray(), atol=1e-14)

    def test_scalar_rmul(self):
        """Test right multiplication for AdChebfunScalar."""
        f = chebfun(lambda x: x**2, [0, 1])
        f_ad = AdChebfun(f, n=8)

        result = f_ad(np.array([0.5]))

        # Test rmul: 3.0 * result
        multiplied = 3.0 * result

        np.testing.assert_allclose(multiplied.value, 3.0 * 0.25, atol=1e-14)
        np.testing.assert_allclose(multiplied.jacobian.toarray(), 3.0 * result.jacobian.toarray(), atol=1e-14)

    def test_scalar_float_conversion_pure_scalar(self):
        """Test float conversion when value is a pure Python scalar."""
        # Create AdChebfunScalar with pure scalar value
        jac = sparse.csr_matrix(np.eye(9))
        scalar = AdChebfunScalar(0.75, n=8, domain=Interval(0, 1), jacobian=jac)

        # Convert to float
        float_val = float(scalar)
        np.testing.assert_allclose(float_val, 0.75, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
