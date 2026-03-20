"""Unit-tests for the chebyshev module (src/chebpy/chebyshev.py).

This single file consolidates every test that exercises the ChebyshevPolynomial
class and its factory functions.
"""

import warnings
from unittest.mock import patch

import numpy as np
import numpy.linalg as la
import numpy.polynomial.chebyshev as cheb
import pytest
from matplotlib import pyplot as plt

from chebpy.algorithms import coeffs2vals2, vals2coeffs2
from chebpy.chebyshev import (
    ChebyshevPolynomial,
    from_coefficients,
    from_constant,
    from_function,
    from_roots,
    from_values,
)
from chebpy.utilities import Interval

# ---------------------------------------------------------------------------
#  Shared / generic plotting tests
# ---------------------------------------------------------------------------
from tests.generic.plotting import test_plot  # noqa: F401

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_coeffs():
    """Fixture for valid coefficients."""
    return [1, 2, 3]


@pytest.fixture
def valid_interval():
    """Fixture for a valid interval."""
    return (-1.0, 1.0)


@pytest.fixture
def constfun():
    """Fixture providing a constant ChebyshevPolynomial for generic tests."""
    return from_coefficients([42.0])


@pytest.fixture
def complexfun():
    """Fixture providing a complex ChebyshevPolynomial for generic tests."""
    return from_coefficients([0, 1])


@pytest.fixture
def plotting_fixtures():
    """Fixture providing various ChebyshevPolynomial objects for plotting tests."""
    simple_poly = from_coefficients([1, 2, 3])
    custom_domain_poly = from_coefficients([1, 2, 3], domain=(0, 2))
    complex_poly = from_coefficients([1 + 1j, 2 + 2j, 3 + 3j])
    return {
        "simple": simple_poly,
        "custom_domain": custom_domain_poly,
        "complex": complex_poly,
    }


# ---------------------------------------------------------------------------
#  Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for ChebyshevPolynomial construction methods."""

    def test_direct_instantiation_with_coefficients(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert isinstance(poly, ChebyshevPolynomial)
        assert np.array_equal(poly.coef, np.array(valid_coeffs))
        assert np.array_equal(poly.domain, np.array(valid_interval))

    def test_direct_instantiation_with_numpy_array_coeffs(self):
        coeffs_np = np.array([4, 5, 6])
        poly = ChebyshevPolynomial(coef=coeffs_np)
        assert isinstance(poly, ChebyshevPolynomial)
        assert np.array_equal(poly.coef, coeffs_np)

    def test_direct_instantiation_with_custom_domain(self):
        domain = (0.0, 2.0)
        poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_direct_instantiation_with_interval_object(self):
        interval = Interval(-2.0, 2.0)
        poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=interval)
        assert np.array_equal(poly.domain, np.array([-2.0, 2.0]))

    def test_from_coefficients_creates_polynomial(self, valid_coeffs, valid_interval):
        poly = from_coefficients(valid_coeffs, valid_interval)
        assert isinstance(poly, ChebyshevPolynomial)
        assert np.array_equal(poly.coef, np.array(valid_coeffs))
        assert np.array_equal(poly.domain, np.array(valid_interval))

    def test_from_coefficients_with_empty_coeffs(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_coefficients([])

    def test_from_coefficients_with_interval_object(self):
        interval = Interval(-2.0, 2.0)
        poly = from_coefficients([1, 2, 3], interval)
        assert np.array_equal(poly.domain, np.array([-2.0, 2.0]))

    def test_from_values_creates_polynomial(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        poly = from_values(values)
        assert isinstance(poly, ChebyshevPolynomial)
        assert poly.coef.size > 0

    def test_from_values_with_domain(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        domain = (0.0, 2.0)
        poly = from_values(values, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_values_with_empty_values(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_values([])

    def test_from_roots_creates_polynomial(self):
        roots = [0.5, -0.5]
        poly = from_roots(roots)
        assert isinstance(poly, ChebyshevPolynomial)
        for root in roots:
            assert abs(poly(root)) < 1e-10

    def test_from_roots_with_domain(self):
        roots = [0.5, -0.5]
        domain = (0.0, 2.0)
        poly = from_roots(roots, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_roots_with_empty_roots(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_roots([])

    def test_from_constant_creates_polynomial(self):
        value = 42.0
        poly = from_constant(value)
        assert isinstance(poly, ChebyshevPolynomial)
        assert poly.coef.size == 1
        assert poly.coef[0] == value
        assert poly.degree() == 0

    def test_from_constant_with_integer_value(self):
        value = 42
        poly = from_constant(value)
        assert isinstance(poly.coef[0], float)
        assert poly.coef[0] == float(value)

    def test_from_constant_with_complex_value(self):
        value = 1 + 2j
        poly = from_constant(value)
        assert poly.iscomplex
        assert poly.coef[0] == value

    def test_from_constant_with_domain(self):
        value = 42.0
        domain = (0.0, 2.0)
        poly = from_constant(value, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_constant_with_non_scalar_value(self):
        with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
            from_constant([1, 2, 3])

    def test_from_function_creates_polynomial(self):
        def fun(x):
            return x**2 + 2 * x + 1

        poly = from_function(fun)
        assert isinstance(poly, ChebyshevPolynomial)
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_with_fixed_degrees_of_freedom(self):
        def fun(x):
            return x**2 + 2 * x + 1

        n = 5
        poly = from_function(fun, n=n)
        assert poly.coef.size <= n
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_with_adaptive_algorithm(self):
        def fun(x):
            return x**2 + 2 * x + 1

        poly = from_function(fun, n=None)
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_with_domain(self):
        def fun(x):
            return x**2 + 2 * x + 1

        domain = (0.0, 2.0)
        poly = from_function(fun, domain=domain)
        assert np.array_equal(poly.domain, np.array(domain))
        x_values = np.linspace(domain[0], domain[1], 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_with_linear_function(self):
        def fun(x):
            return 2 * x + 1

        poly = from_function(fun)
        assert np.sum(np.abs(poly.coef) > 1e-10) <= 2
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_with_complex_function(self):
        def fun(x):
            return (1 + 2j) * x**2 + (3 + 4j) * x + (5 + 6j)

        poly = from_function(fun)
        assert poly.iscomplex
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(poly(x) - fun(x)) < 1e-10

    def test_from_function_non_convergence_warning(self):
        def step_function(x):
            return np.where(x > 0, 1.0, -1.0)

        with patch("chebpy.chebyshev.prefs") as mock_prefs:
            mock_prefs.maxpow2 = 4
            mock_prefs.eps = 2.220446049250313e-16
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                poly = from_function(step_function)
                assert isinstance(poly, ChebyshevPolynomial)
                assert len(w) == 1
                assert "did not converge" in str(w[0].message)


# ---------------------------------------------------------------------------
#  Class usage
# ---------------------------------------------------------------------------


class TestClassUsage:
    """Tests for ChebyshevPolynomial class usage."""

    def test_post_init_converts_coeffs_to_numpy_array(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert isinstance(poly.coef, np.ndarray)

    def test_roots_with_valid_coefficients(self, valid_coeffs):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=(-1.0, 1.0))
        roots = poly.roots()
        assert isinstance(roots, np.ndarray)
        assert len(roots) <= poly.degree()

    def test_degree_correctly_calculates(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert poly.degree() == len(valid_coeffs) - 1

    def test_eq_returns_true_for_equal_polynomials(self, valid_coeffs, valid_interval):
        poly1 = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        poly2 = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert poly1 == poly2

    def test_eq_returns_false_for_different_coeffs(self, valid_interval):
        poly1 = ChebyshevPolynomial(coef=[1, 2, 3], domain=valid_interval)
        poly2 = ChebyshevPolynomial(coef=[4, 5, 6], domain=valid_interval)
        assert poly1 != poly2

    def test_eq_returns_false_for_different_domains(self, valid_coeffs):
        poly1 = ChebyshevPolynomial(coef=valid_coeffs, domain=(-1.0, 1.0))
        poly2 = ChebyshevPolynomial(coef=valid_coeffs, domain=(0.0, 1.0))
        assert poly1 != poly2

    def test_eq_returns_false_for_non_chebyshev_objects(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert poly != "not a polynomial"

    def test_call_evaluates_at_single_point(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        result = poly(0.5)
        assert isinstance(result, float)

    def test_call_evaluates_at_multiple_points(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        points = np.linspace(-1.0, 1.0, 10)
        results = poly(points)
        assert isinstance(results, np.ndarray)
        assert results.shape == points.shape

    def test_call_handles_out_of_domain_points(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        result = poly(2.0)
        assert isinstance(result, float)

    def test_call_handles_empty_input(self, valid_coeffs):
        poly = ChebyshevPolynomial(coef=valid_coeffs)
        results = poly(np.array([]))
        assert isinstance(results, np.ndarray)
        assert results.size == 0

    def test_post_init_converts_interval(self, valid_coeffs, valid_interval):
        poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
        assert isinstance(poly.domain, np.ndarray)

    def test_post_init_raises_error_on_invalid_interval(self):
        with pytest.raises(TypeError):
            ChebyshevPolynomial()

    def test_post_init_handles_numpy_array_coeffs(self):
        coeffs_np = np.array([4, 5, 6])
        poly = ChebyshevPolynomial(coef=coeffs_np)
        assert np.array_equal(poly.coef, coeffs_np)

    def test_iscomplex_property_real(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        assert not poly.iscomplex

    def test_iscomplex_property_complex(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        assert poly.iscomplex

    def test_copy_method(self):
        coef = [1, 2, 3]
        domain = (0.0, 2.0)
        symbol = "t"
        poly = ChebyshevPolynomial(coef=coef, domain=domain, symbol=symbol)
        poly_copy = poly.copy()
        assert poly_copy is not poly
        assert np.array_equal(poly_copy.coef, poly.coef)
        assert np.array_equal(poly_copy.domain, poly.domain)
        assert np.array_equal(poly_copy.window, poly.window)
        assert poly_copy.symbol == poly.symbol
        poly_copy.coef[0] = 999
        assert poly.coef[0] != 999

    def test_diff_returns_chebyshev_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        deriv = poly.diff()
        assert isinstance(deriv, ChebyshevPolynomial)

    def test_diff_preserves_domain_and_window(self):
        domain = (0.0, 2.0)
        poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=domain)
        deriv = poly.diff()
        assert np.array_equal(deriv.domain, poly.domain)
        assert np.array_equal(deriv.window, poly.window)

    def test_cumsum_returns_chebyshev_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        integ = poly.cumsum()
        assert isinstance(integ, ChebyshevPolynomial)

    def test_cumsum_preserves_domain_and_window(self):
        domain = (0.0, 2.0)
        poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=domain)
        integ = poly.cumsum()
        assert np.array_equal(integ.domain, poly.domain)
        assert np.array_equal(integ.window, poly.window)

    def test_values_property(self):
        coef = [1, 2, 3]
        poly = ChebyshevPolynomial(coef=coef)
        values = poly.values
        assert isinstance(values, np.ndarray)
        assert len(values) == len(coef)
        expected_values = coeffs2vals2(np.array(coef))
        assert np.allclose(values, expected_values)
        reconstructed_coef = vals2coeffs2(values)
        assert np.allclose(reconstructed_coef, coef)

    def test_roundtrip_from_values_to_values(self):
        original_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        poly = from_values(original_values)
        retrieved_values = poly.values
        assert np.allclose(retrieved_values, original_values)
        poly_with_domain = from_values(original_values, domain=(0.0, 2.0))
        assert np.allclose(poly_with_domain.values, original_values)

    def test_prolong_truncation(self):
        coef = [1, 2, 3, 4, 5]
        poly = ChebyshevPolynomial(coef=coef)
        n = 3
        truncated = poly.prolong(n)
        assert truncated.size == n
        assert np.array_equal(truncated.coef, np.array(coef[:n]))
        assert poly.size == len(coef)

    def test_prolong_zero_padding(self):
        coef = [1, 2, 3]
        poly = ChebyshevPolynomial(coef=coef)
        n = 5
        padded = poly.prolong(n)
        assert padded.size == n
        expected_coef = np.concatenate([np.array(coef), np.zeros(n - len(coef))])
        assert np.array_equal(padded.coef, expected_coef)

    def test_prolong_copy(self):
        coef = [1, 2, 3]
        poly = ChebyshevPolynomial(coef=coef)
        n = len(coef)
        copy_poly = poly.prolong(n)
        assert copy_poly is not poly
        assert copy_poly.size == n
        assert np.array_equal(copy_poly.coef, poly.coef)
        copy_poly.coef[0] = 999
        assert poly.coef[0] != 999

    def test_prolong_preserves_attributes(self):
        coef = [1, 2, 3]
        domain = (0.0, 2.0)
        symbol = "t"
        poly = ChebyshevPolynomial(coef=coef, domain=domain, symbol=symbol)
        for n in [2, 3, 4]:
            result = poly.prolong(n)
            assert np.array_equal(result.domain, poly.domain)
            assert np.array_equal(result.window, poly.window)
            assert result.symbol == poly.symbol

    def test_isconst_with_constant_polynomial(self):
        poly = ChebyshevPolynomial(coef=[42.0])
        assert poly.isconst

    def test_isconst_with_non_constant_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        assert not poly.isconst

    def test_isconst_with_empty_polynomial(self):
        with pytest.raises(ValueError, match="Coefficient array is empty"):
            ChebyshevPolynomial(coef=[])

    def test_vscale_with_constant_polynomial(self):
        value = 42.0
        poly = ChebyshevPolynomial(coef=[value])
        assert poly.vscale == abs(value)

    def test_vscale_with_simple_polynomial(self):
        coef = [1, 2, 3]
        poly = ChebyshevPolynomial(coef=coef)
        values = coeffs2vals2(np.array(coef))
        expected_vscale = np.abs(values).max()
        assert np.isclose(poly.vscale, expected_vscale)

    def test_vscale_with_complex_polynomial(self):
        coef = [1 + 1j, 2 + 2j, 3 + 3j]
        poly = ChebyshevPolynomial(coef=coef)
        values = coeffs2vals2(np.array(coef))
        expected_vscale = np.abs(values).max()
        assert np.isclose(poly.vscale, expected_vscale)

    def test_vscale_with_empty_polynomial(self):
        with pytest.raises(ValueError, match="Coefficient array is empty"):
            ChebyshevPolynomial(coef=[])

    def test_sum_method(self):
        poly = ChebyshevPolynomial(coef=[1.0])
        assert np.isclose(poly.sum(), 2.0)
        poly_x = ChebyshevPolynomial(coef=[0.0, 1.0])
        assert np.isclose(poly_x.sum(), 0.0, atol=1e-14)
        poly_domain = ChebyshevPolynomial(coef=[1.0], domain=(0, 2))
        assert np.isclose(poly_domain.sum(), 2.0)

    # Factory function tests also from class_usage.py
    def test_from_coefficients_class_usage(self, valid_coeffs, valid_interval):
        poly = from_coefficients(valid_coeffs, valid_interval)
        assert isinstance(poly, ChebyshevPolynomial)
        assert np.array_equal(poly.coef, np.array(valid_coeffs))
        assert np.array_equal(poly.domain, np.array(valid_interval))

    def test_from_coefficients_with_empty_coeffs_class_usage(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_coefficients([])

    def test_from_coefficients_with_interval_object_class_usage(self):
        interval = Interval(-2.0, 2.0)
        poly = from_coefficients([1, 2, 3], interval)
        assert np.array_equal(poly.domain, np.array([-2.0, 2.0]))

    def test_from_values_class_usage(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        poly = from_values(values)
        assert isinstance(poly, ChebyshevPolynomial)

    def test_from_values_with_domain_class_usage(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        domain = (0.0, 2.0)
        poly = from_values(values, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_values_with_empty_values_class_usage(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_values([])

    def test_from_roots_class_usage(self):
        roots = [0.5, -0.5]
        poly = from_roots(roots)
        for root in roots:
            assert abs(poly(root)) < 1e-10

    def test_from_roots_with_domain_class_usage(self):
        roots = [0.5, -0.5]
        domain = (0.0, 2.0)
        poly = from_roots(roots, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_roots_with_empty_roots_class_usage(self):
        with pytest.raises(ValueError, match=r"\[\]"):
            from_roots([])

    def test_round_trip_from_roots(self):
        original_poly = ChebyshevPolynomial(coef=[1, 0, -1])
        roots = original_poly.roots()
        recreated_poly = from_roots(roots)
        original_normalized = ChebyshevPolynomial(coef=original_poly.coef / original_poly.coef[0])
        recreated_normalized = ChebyshevPolynomial(coef=recreated_poly.coef / recreated_poly.coef[0])
        assert original_normalized == recreated_normalized

    def test_from_constant_class_usage(self):
        value = 42.0
        poly = from_constant(value)
        assert poly.coef.size == 1
        assert poly.coef[0] == value

    def test_from_constant_with_integer_class_usage(self):
        value = 42
        poly = from_constant(value)
        assert isinstance(poly.coef[0], float)

    def test_from_constant_with_complex_value_class_usage(self):
        value = 3 + 4j
        poly = from_constant(value)
        assert poly.iscomplex
        assert poly.coef[0] == value

    def test_from_constant_with_domain_class_usage(self):
        value = 42.0
        domain = (0.0, 2.0)
        poly = from_constant(value, domain)
        assert np.array_equal(poly.domain, np.array(domain))

    def test_from_constant_with_interval_object_class_usage(self):
        value = 42.0
        interval = Interval(-2.0, 2.0)
        poly = from_constant(value, interval)
        assert np.array_equal(poly.domain, np.array([-2.0, 2.0]))

    def test_from_constant_with_non_scalar_class_usage(self):
        with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
            from_constant([1, 2, 3])

    def test_constant_polynomial_evaluation(self):
        value = 42.0
        poly = from_constant(value)
        points = np.linspace(-1.0, 1.0, 10)
        results = poly(points)
        assert np.allclose(results, value)


# ---------------------------------------------------------------------------
#  Arithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    """Tests for ChebyshevPolynomial arithmetic operations."""

    def test_add_polynomials(self):
        poly1 = ChebyshevPolynomial(coef=[1, 2, 3])
        poly2 = ChebyshevPolynomial(coef=[4, 5, 6])
        result = poly1 + poly2
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [5, 7, 9])
        assert np.array_equal(result.domain, poly1.domain)
        assert np.array_equal(result.window, poly1.window)

    def test_add_polynomial_and_scalar(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = poly + 4
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [5, 2, 3])
        assert np.array_equal(result.domain, poly.domain)

    def test_radd_scalar_and_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = 4 + poly
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [5, 2, 3])

    def test_subtract_polynomials(self):
        poly1 = ChebyshevPolynomial(coef=[4, 5, 6])
        poly2 = ChebyshevPolynomial(coef=[1, 2, 3])
        result = poly1 - poly2
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [3, 3, 3])

    def test_subtract_polynomial_and_scalar(self):
        poly = ChebyshevPolynomial(coef=[4, 5, 6])
        result = poly - 1
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [3, 5, 6])

    def test_rsubtract_scalar_and_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = 4 - poly
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [3, -2, -3])

    def test_multiply_polynomials(self):
        poly1 = ChebyshevPolynomial(coef=[1, 2])
        poly2 = ChebyshevPolynomial(coef=[3, 4])
        result = poly1 * poly2
        assert isinstance(result, ChebyshevPolynomial)
        expected_coef = [7, 10, 4]
        assert np.allclose(result.coef, expected_coef)

    def test_multiply_polynomial_and_scalar(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = poly * 2
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [2, 4, 6])

    def test_rmultiply_scalar_and_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = 2 * poly
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [2, 4, 6])

    def test_divide_polynomial_by_scalar(self):
        poly = ChebyshevPolynomial(coef=[2, 4, 6])
        result = poly / 2
        assert isinstance(result, ChebyshevPolynomial)
        assert np.array_equal(result.coef, [1, 2, 3])

    def test_power_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 1])
        result = poly**2
        assert isinstance(result, ChebyshevPolynomial)
        expected_coef = [1.5, 2, 0.5]
        assert np.allclose(result.coef, expected_coef)

    def test_evaluate_polynomial_at_points(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        points = np.array([-1.0, 0.0, 1.0])
        results = poly(points)
        expected_results = np.array([2.0, -2.0, 6.0])
        assert np.allclose(results, expected_results)

    def test_evaluate_polynomial_at_point(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        result = poly(0.5)
        assert np.isclose(result, 0.5)
        assert np.isscalar(result)

    def test_evaluate_derivative_at_point(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        deriv = poly.diff()
        result = deriv(0.5)
        assert np.isclose(result, 8.0)
        assert np.isscalar(result)


# ---------------------------------------------------------------------------
#  Calculus
# ---------------------------------------------------------------------------


class TestCalculus:
    """Tests for ChebyshevPolynomial calculus operations."""

    def test_diff_calculates_derivative_correctly(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        deriv = poly.diff()
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            expected = 2 + 12 * x
            actual = deriv(x)
            assert abs(actual - expected) < 1e-10

    def test_diff_of_constant_polynomial(self):
        poly = from_constant(42.0)
        deriv = poly.diff()
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(deriv(x)) < 1e-10

    def test_cumsum_calculates_antiderivative_correctly(self):
        poly = ChebyshevPolynomial(coef=[0, 1])  # T_1(x) = x
        integ = poly.cumsum()
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            expected = (x**2 - 1) / 2
            actual = integ(x)
            assert abs(actual - expected) < 1e-10

    def test_cumsum_of_zero_polynomial(self):
        poly = ChebyshevPolynomial(coef=[0])
        integ = poly.cumsum()
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            assert abs(integ(x)) < 1e-10

    def test_round_trip_diff_cumsum(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        round_trip = poly.cumsum().diff()
        x_values = np.linspace(-1, 1, 10)
        for x in x_values:
            original = poly(x)
            result = round_trip(x)
            assert abs(result - original) < 1e-10


# ---------------------------------------------------------------------------
#  Complex
# ---------------------------------------------------------------------------


class TestComplex:
    """Tests for complex ChebyshevPolynomial functionality."""

    def test_iscomplex_with_real(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        assert not poly.iscomplex

    def test_iscomplex_with_complex(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        assert poly.iscomplex

    def test_real_method_with_real_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        real_poly = poly.real()
        assert real_poly is poly
        assert np.array_equal(real_poly.coef, poly.coef)

    def test_real_method_with_complex_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        real_poly = poly.real()
        assert real_poly is not poly
        assert np.array_equal(real_poly.coef, np.real(poly.coef))
        assert np.array_equal(real_poly.domain, poly.domain)
        assert np.array_equal(real_poly.window, poly.window)
        assert real_poly.symbol == poly.symbol

    def test_imag_method_with_real_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        imag_poly = poly.imag()
        assert imag_poly is poly
        assert np.array_equal(imag_poly.coef, poly.coef)

    def test_imag_method_with_complex_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        imag_poly = poly.imag()
        assert imag_poly is not poly
        assert np.array_equal(imag_poly.coef, np.imag(poly.coef))
        assert np.array_equal(imag_poly.domain, poly.domain)
        assert imag_poly.symbol == poly.symbol

    def test_evaluate_complex_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        x = 0.5
        result = poly(x)
        expected_real = 1 + 2 * x + 3 * (2 * x**2 - 1)
        expected_imag = 1 + 2 * x + 3 * (2 * x**2 - 1)
        expected = complex(expected_real, expected_imag)
        assert np.isclose(result, expected)

    def test_real_plus_imag_equals_original(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        real_poly = poly.real()
        imag_poly = poly.imag()
        reconstructed = real_poly + 1j * imag_poly
        assert np.allclose(reconstructed.coef, poly.coef)
        assert np.array_equal(reconstructed.domain, poly.domain)

    def test_real_and_imag_evaluate_correctly(self):
        poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
        real_poly = poly.real()
        imag_poly = poly.imag()
        points = np.linspace(-1, 1, 10)
        for x in points:
            result = poly(x)
            assert np.isclose(real_poly(x), np.real(result))
            assert np.isclose(imag_poly(x), np.imag(result))


# ---------------------------------------------------------------------------
#  Roots
# ---------------------------------------------------------------------------


class TestRoots:
    """Tests for ChebyshevPolynomial roots, companion matrices, and eigenvalues."""

    def test_roots_of_quadratic_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 0, -1])
        roots = poly.roots()
        assert len(roots) == 2
        assert np.allclose(sorted(roots), [-1, 1])

    def test_roots_of_cubic_polynomial(self):
        poly = from_roots([-1, 0, 1])
        roots = poly.roots()
        assert len(roots) == 3
        assert np.allclose(sorted(roots), [-1, 0, 1])

    def test_roots_of_polynomial_with_multiple_roots(self):
        poly = from_roots([0, 0, 1])
        roots = poly.roots()
        assert len(roots) == 3
        sorted_roots = sorted(roots)
        assert np.isclose(sorted_roots[0], 0, atol=1e-7)
        assert np.isclose(sorted_roots[1], 0, atol=1e-7)
        assert np.isclose(sorted_roots[2], 1, atol=1e-7)

    def test_roots_of_polynomial_with_complex_roots(self):
        poly = from_roots([1 + 1j, 1 - 1j])
        roots = poly.roots()
        assert len(roots) == 2
        expected_roots = [1 + 1j, 1 - 1j]
        for root in roots:
            assert any(np.isclose(root, expected_root, atol=1e-7) for expected_root in expected_roots)

    def test_companion_matrix_of_quadratic_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3])
        companion = cheb.chebcompanion(poly.coef)
        assert companion.shape == (poly.degree(), poly.degree())
        eigenvalues = la.eigvals(companion)
        roots = poly.roots()
        eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
        roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))
        assert np.allclose(eigenvalues, roots)

    def test_companion_matrix_of_cubic_polynomial(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3, 4])
        companion = cheb.chebcompanion(poly.coef)
        assert companion.shape == (poly.degree(), poly.degree())
        eigenvalues = la.eigvals(companion)
        roots = poly.roots()
        eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
        roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))
        assert np.allclose(eigenvalues, roots)

    def test_eigenvalues_of_companion_matrix(self):
        roots_expected = [-1, 0, 1]
        poly = from_roots(roots_expected)
        companion = cheb.chebcompanion(poly.coef)
        eigenvalues = la.eigvals(companion)
        eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
        roots_expected = sorted(roots_expected)
        assert np.allclose(eigenvalues, roots_expected)

    def test_eigenvalues_of_companion_matrix_with_complex_roots(self):
        roots_expected = [1 + 1j, 1 - 1j]
        poly = from_roots(roots_expected)
        companion = cheb.chebcompanion(poly.coef)
        eigenvalues = la.eigvals(companion)
        for eigenvalue in eigenvalues:
            assert any(np.isclose(eigenvalue, root_expected, atol=1e-7) for root_expected in roots_expected)

    def test_roots_companion_eigenvalues_relationship(self):
        poly = ChebyshevPolynomial(coef=[1, 2, 3, 4, 5])
        roots = poly.roots()
        companion = cheb.chebcompanion(poly.coef)
        eigenvalues = la.eigvals(companion)
        roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))
        eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
        assert np.allclose(roots, eigenvalues)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    """Tests for ChebyshevPolynomial plotting methods."""

    def test_plot_with_options(self, plotting_fixtures):
        _fig, ax = plt.subplots()
        poly = plotting_fixtures["simple"]
        poly.plot(ax=ax, color="r", linestyle="--", linewidth=2)

    def test_plot_custom_domain(self, plotting_fixtures):
        _fig, ax = plt.subplots()
        poly = plotting_fixtures["custom_domain"]
        poly.plot(ax=ax)

    def test_plot_complex_polynomial(self, plotting_fixtures):
        _fig, ax = plt.subplots()
        poly = plotting_fixtures["complex"]
        poly.plot(ax=ax)
