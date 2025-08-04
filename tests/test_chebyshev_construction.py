"""Tests for the construction of Chebyshev polynomials."""

import numpy as np
import pytest

from chebpy.core.chebyshev import (
    ChebyshevPolynomial,
    from_coefficients,
    from_constant,
    from_function,
    from_roots,
    from_values,
)
from chebpy.core.utilities import Interval


@pytest.fixture
def valid_coeffs():
    """Fixture for valid coefficients."""
    return [1, 2, 3]


@pytest.fixture
def valid_interval():
    """Fixture for a valid interval."""
    return (-1.0, 1.0)


# Tests for direct instantiation
def test_direct_instantiation_with_coefficients(valid_coeffs, valid_interval):
    """Test direct instantiation of ChebyshevPolynomial with coefficients."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert isinstance(poly, ChebyshevPolynomial), "Direct instantiation did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.coef, np.array(valid_coeffs)), "Coefficients were not set correctly."
    assert np.array_equal(poly.domain, np.array(valid_interval)), "Domain was not set correctly."


def test_direct_instantiation_with_numpy_array_coeffs():
    """Test direct instantiation with numpy array coefficients."""
    coeffs_np = np.array([4, 5, 6])
    poly = ChebyshevPolynomial(coef=coeffs_np)
    assert isinstance(poly, ChebyshevPolynomial), "Direct instantiation did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.coef, coeffs_np), "Coefficients were not set correctly."


def test_direct_instantiation_with_custom_domain():
    """Test direct instantiation with a custom domain."""
    domain = (0.0, 2.0)
    poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=domain)
    assert isinstance(poly, ChebyshevPolynomial), "Direct instantiation did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array(domain)), "Domain was not set correctly."


def test_direct_instantiation_with_interval_object():
    """Test direct instantiation with an Interval object."""
    interval = Interval(-2.0, 2.0)
    poly = ChebyshevPolynomial(coef=[1, 2, 3], domain=interval)
    assert isinstance(poly, ChebyshevPolynomial), "Direct instantiation did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array([-2.0, 2.0])), "Domain was not set correctly."


# Tests for from_coefficients factory function
def test_from_coefficients_creates_chebyshev_polynomial(valid_coeffs, valid_interval):
    """Test that from_coefficients creates a ChebyshevPolynomial with the given coefficients."""
    poly = from_coefficients(valid_coeffs, valid_interval)
    assert isinstance(poly, ChebyshevPolynomial), "from_coefficients did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.coef, np.array(valid_coeffs)), "Coefficients were not set correctly."
    assert np.array_equal(poly.domain, np.array(valid_interval)), "Domain was not set correctly."


def test_from_coefficients_with_empty_coeffs():
    """Test that from_coefficients works with empty coefficients."""
    with pytest.raises(ValueError):
        from_coefficients([])


def test_from_coefficients_with_interval_object():
    """Test that from_coefficients works with an Interval object."""
    interval = Interval(-2.0, 2.0)
    poly = from_coefficients([1, 2, 3], interval)
    assert isinstance(poly, ChebyshevPolynomial), "from_coefficients did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array([-2.0, 2.0])), "Domain was not set correctly."


# Tests for from_values factory function
def test_from_values_creates_chebyshev_polynomial():
    """Test that from_values creates a ChebyshevPolynomial from values at Chebyshev points."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    poly = from_values(values)
    assert isinstance(poly, ChebyshevPolynomial), "from_values did not return a ChebyshevPolynomial."
    assert poly.coef.size > 0, "Coefficients were not computed correctly."


def test_from_values_with_domain():
    """Test that from_values works with a custom domain."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    domain = (0.0, 2.0)
    poly = from_values(values, domain)
    assert isinstance(poly, ChebyshevPolynomial), "from_values did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array(domain)), "Domain was not set correctly."


def test_from_values_with_empty_values():
    """Test that from_values works with empty values."""
    with pytest.raises(ValueError):
        from_values([])


# Tests for from_roots factory function
def test_from_roots_creates_chebyshev_polynomial():
    """Test that from_roots creates a ChebyshevPolynomial with the given roots."""
    roots = [0.5, -0.5]
    poly = from_roots(roots)
    assert isinstance(poly, ChebyshevPolynomial), "from_roots did not return a ChebyshevPolynomial."

    # Check that the polynomial evaluates to zero at the roots
    for root in roots:
        assert abs(poly(root)) < 1e-10, f"Polynomial does not evaluate to zero at root {root}."


def test_from_roots_with_domain():
    """Test that from_roots works with a custom domain."""
    roots = [0.5, -0.5]
    domain = (0.0, 2.0)
    poly = from_roots(roots, domain)
    assert isinstance(poly, ChebyshevPolynomial), "from_roots did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array(domain)), "Domain was not set correctly."


def test_from_roots_with_empty_roots():
    """Test that from_roots works with empty roots."""
    with pytest.raises(ValueError):
        from_roots([])


# Tests for from_constant factory function
def test_from_constant_creates_chebyshev_polynomial():
    """Test that from_constant creates a ChebyshevPolynomial with the given constant value."""
    value = 42.0
    poly = from_constant(value)
    assert isinstance(poly, ChebyshevPolynomial), "from_constant did not return a ChebyshevPolynomial."
    assert poly.coef.size == 1, "Polynomial should have exactly one coefficient."
    assert poly.coef[0] == value, "Coefficient value does not match the input constant."
    assert poly.degree() == 0, "Constant polynomial should have degree 0."


def test_from_constant_with_integer_value():
    """Test that from_constant works with integer values."""
    value = 42
    poly = from_constant(value)
    assert isinstance(poly, ChebyshevPolynomial), "from_constant did not return a ChebyshevPolynomial."
    assert poly.coef.size == 1, "Polynomial should have exactly one coefficient."
    assert poly.coef[0] == float(value), "Integer value was not converted to float."


def test_from_constant_with_complex_value():
    """Test that from_constant works with complex values."""
    value = 1 + 2j
    poly = from_constant(value)
    assert isinstance(poly, ChebyshevPolynomial), "from_constant did not return a ChebyshevPolynomial."
    assert poly.coef.size == 1, "Polynomial should have exactly one coefficient."
    assert poly.coef[0] == value, "Coefficient value does not match the input constant."
    assert poly.iscomplex, "Polynomial with complex coefficient should have iscomplex=True."


def test_from_constant_with_domain():
    """Test that from_constant works with a custom domain."""
    value = 42.0
    domain = (0.0, 2.0)
    poly = from_constant(value, domain)
    assert isinstance(poly, ChebyshevPolynomial), "from_constant did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array(domain)), "Domain was not set correctly."


def test_from_constant_with_non_scalar_value():
    """Test that from_constant raises an error for non-scalar values."""
    with pytest.raises(ValueError):
        from_constant([1, 2, 3])


# Tests for from_function factory function
def test_from_function_creates_chebyshev_polynomial():
    """Test that from_function creates a ChebyshevPolynomial from a callable function."""

    def fun(x):
        return x**2 + 2 * x + 1  # Quadratic function

    poly = from_function(fun)
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."
    assert poly.coef.size > 0, "Coefficients were not computed correctly."

    # Check that the polynomial approximates the function
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."


def test_from_function_with_fixed_degrees_of_freedom():
    """Test that from_function works with a fixed number of degrees of freedom."""

    def fun(x):
        return x**2 + 2 * x + 1  # Quadratic function

    n = 5  # Fixed number of degrees of freedom
    poly = from_function(fun, n=n)
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."
    assert poly.coef.size <= n, f"Polynomial has more than {n} coefficients."

    # Check that the polynomial approximates the function
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."


def test_from_function_with_adaptive_algorithm():
    """Test that from_function works with the adaptive algorithm."""

    def fun(x):
        return x**2 + 2 * x + 1  # Quadratic function

    poly = from_function(fun, n=None)  # Use adaptive algorithm
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."
    assert poly.coef.size > 0, "Coefficients were not computed correctly."

    # Check that the polynomial approximates the function
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."


def test_from_function_with_domain():
    """Test that from_function works with a custom domain."""

    def fun(x):
        return x**2 + 2 * x + 1  # Quadratic function

    domain = (0.0, 2.0)
    poly = from_function(fun, domain=domain)
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."
    assert np.array_equal(poly.domain, np.array(domain)), "Domain was not set correctly."

    # Check that the polynomial approximates the function on the custom domain
    x_values = np.linspace(domain[0], domain[1], 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."


def test_from_function_with_linear_function():
    """Test that from_function works with a linear function."""

    def fun(x):
        return 2 * x + 1  # Linear function

    poly = from_function(fun)
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."

    # For a linear function, we expect at most 2 non-zero coefficients
    assert np.sum(np.abs(poly.coef) > 1e-10) <= 2, "Polynomial has more non-zero coefficients than expected."

    # Check that the polynomial approximates the function
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."


def test_from_function_with_complex_function():
    """Test that from_function works with a complex-valued function."""

    def fun(x):
        return (1 + 2j) * x**2 + (3 + 4j) * x + (5 + 6j)  # Complex quadratic function

    poly = from_function(fun)
    assert isinstance(poly, ChebyshevPolynomial), "from_function did not return a ChebyshevPolynomial."
    assert poly.iscomplex, "Polynomial with complex coefficients should have iscomplex=True."

    # Check that the polynomial approximates the function
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(poly(x) - fun(x)) < 1e-10, f"Polynomial does not approximate the function at x={x}."
