"""Tests for the arithmetic operations of ChebyshevPolynomial."""

import numpy as np

from chebpy.core.chebyshev import ChebyshevPolynomial


def test_add_polynomials():
    """Test addition of two ChebyshevPolynomial objects."""
    # Create two polynomials
    poly1 = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)
    poly2 = ChebyshevPolynomial(coef=[4, 5, 6])  # 4 + 5*T_1(x) + 6*T_2(x)

    # Add them
    result = poly1 + poly2

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct
    assert np.array_equal(result.coef, [5, 7, 9]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly1.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly1.window), "Window was not preserved"


def test_add_polynomial_and_scalar():
    """Test addition of a ChebyshevPolynomial and a scalar."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Add a scalar
    scalar = 4
    result = poly + scalar

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (scalar is added to the constant term)
    assert np.array_equal(result.coef, [5, 2, 3]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_radd_scalar_and_polynomial():
    """Test right addition of a scalar and a ChebyshevPolynomial."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Add a scalar on the right
    scalar = 4
    result = scalar + poly

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (scalar is added to the constant term)
    assert np.array_equal(result.coef, [5, 2, 3]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_subtract_polynomials():
    """Test subtraction of two ChebyshevPolynomial objects."""
    # Create two polynomials
    poly1 = ChebyshevPolynomial(coef=[4, 5, 6])  # 4 + 5*T_1(x) + 6*T_2(x)
    poly2 = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Subtract them
    result = poly1 - poly2

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct
    assert np.array_equal(result.coef, [3, 3, 3]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly1.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly1.window), "Window was not preserved"


def test_subtract_polynomial_and_scalar():
    """Test subtraction of a ChebyshevPolynomial and a scalar."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[4, 5, 6])  # 4 + 5*T_1(x) + 6*T_2(x)

    # Subtract a scalar
    scalar = 1
    result = poly - scalar

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (scalar is subtracted from the constant term)
    assert np.array_equal(result.coef, [3, 5, 6]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_rsubtract_scalar_and_polynomial():
    """Test right subtraction of a scalar and a ChebyshevPolynomial."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Subtract polynomial from scalar
    scalar = 4
    result = scalar - poly

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (polynomial is subtracted from scalar)
    assert np.array_equal(result.coef, [3, -2, -3]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_multiply_polynomials():
    """Test multiplication of two ChebyshevPolynomial objects."""
    # Create two polynomials
    poly1 = ChebyshevPolynomial(coef=[1, 2])  # 1 + 2*T_1(x)
    poly2 = ChebyshevPolynomial(coef=[3, 4])  # 3 + 4*T_1(x)

    # Multiply them
    result = poly1 * poly2

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct
    # For Chebyshev polynomials, the multiplication is more complex than for standard polynomials
    # For T_0(x) * T_0(x) = T_0(x)
    # For T_0(x) * T_1(x) = T_1(x)
    # For T_1(x) * T_1(x) = (T_0(x) + T_2(x))/2
    # So (1 + 2*T_1(x)) * (3 + 4*T_1(x)) = 3 + 4*T_1(x) + 6*T_1(x) + 8*T_1(x)*T_1(x)
    # = 3 + 10*T_1(x) + 8*(T_0(x) + T_2(x))/2 = 3 + 4 + 10*T_1(x) + 4*T_2(x) = 7 + 10*T_1(x) + 4*T_2(x)
    expected_coef = [7, 10, 4]
    assert np.allclose(result.coef, expected_coef), (
        f"Coefficients are not correct. Expected {expected_coef}, got {result.coef}"
    )

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly1.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly1.window), "Window was not preserved"


def test_multiply_polynomial_and_scalar():
    """Test multiplication of a ChebyshevPolynomial and a scalar."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Multiply by a scalar
    scalar = 2
    result = poly * scalar

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (all coefficients are multiplied by the scalar)
    assert np.array_equal(result.coef, [2, 4, 6]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_rmultiply_scalar_and_polynomial():
    """Test right multiplication of a scalar and a ChebyshevPolynomial."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Multiply scalar by polynomial
    scalar = 2
    result = scalar * poly

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (all coefficients are multiplied by the scalar)
    assert np.array_equal(result.coef, [2, 4, 6]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_divide_polynomial_by_scalar():
    """Test division of a ChebyshevPolynomial by a scalar."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[2, 4, 6])  # 2 + 4*T_1(x) + 6*T_2(x)

    # Divide by a scalar
    scalar = 2
    result = poly / scalar

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct (all coefficients are divided by the scalar)
    assert np.array_equal(result.coef, [1, 2, 3]), "Coefficients are not correct"

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_power_polynomial():
    """Test raising a ChebyshevPolynomial to a power."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 1])  # 1 + T_1(x) = 1 + x

    # Raise to a power
    power = 2
    result = poly**power

    # Check that the result is a ChebyshevPolynomial
    assert isinstance(result, ChebyshevPolynomial), "Result is not a ChebyshevPolynomial"

    # Check that the coefficients are correct
    # For (1 + x)^2 = 1 + 2x + x^2, which in Chebyshev basis is 1.5 + 2*T_1(x) + 0.5*T_2(x)
    expected_coef = [1.5, 2, 0.5]
    assert np.allclose(result.coef, expected_coef), (
        f"Coefficients are not correct. Expected {expected_coef}, got {result.coef}"
    )

    # Check that the domain and window are preserved
    assert np.array_equal(result.domain, poly.domain), "Domain was not preserved"
    assert np.array_equal(result.window, poly.window), "Window was not preserved"


def test_evaluate_polynomial_at_points():
    """Test evaluating a ChebyshevPolynomial at multiple points."""
    # Create a polynomial
    poly = ChebyshevPolynomial(
        coef=[1, 2, 3]
    )  # 1 + 2*T_1(x) + 3*T_2(x) = 1 + 2x + 3(2x^2-1) = 1 + 2x + 6x^2 - 3 = -2 + 2x + 6x^2

    # Evaluate at multiple points
    points = np.array([-1.0, 0.0, 1.0])
    results = poly(points)

    # Check that the results are correct
    # At x = -1: -2 + 2*(-1) + 6*(-1)^2 = -2 - 2 + 6 = 2
    # At x = 0: -2 + 2*0 + 6*0^2 = -2
    # At x = 1: -2 + 2*1 + 6*1^2 = -2 + 2 + 6 = 6
    expected_results = np.array([2.0, -2.0, 6.0])
    assert np.allclose(results, expected_results), (
        f"Results are not correct. Expected {expected_results}, got {results}"
    )


def test_evaluate_polynomial_at_point():
    """Test evaluating a ChebyshevPolynomial at a single point."""
    # Create a polynomial
    poly = ChebyshevPolynomial(
        coef=[1, 2, 3]
    )  # 1 + 2*T_1(x) + 3*T_2(x) = 1 + 2x + 3(2x^2-1) = 1 + 2x + 6x^2 - 3 = -2 + 2x + 6x^2

    # Evaluate at a single point
    point = 0.5
    result = poly(point)

    # Check that the result is correct
    # At x = 0.5: -2 + 2*0.5 + 6*0.5^2 = -2 + 1 + 1.5 = 0.5
    expected_result = 0.5
    assert np.isclose(result, expected_result), f"Result is not correct. Expected {expected_result}, got {result}"

    # Check that the result is a scalar, not a ChebyshevPolynomial
    assert np.isscalar(result), "Result is not a scalar"


def test_evaluate_derivative_at_point():
    """Test evaluating the derivative of a ChebyshevPolynomial at a single point."""
    # Create a polynomial
    poly = ChebyshevPolynomial(
        coef=[1, 2, 3]
    )  # 1 + 2*T_1(x) + 3*T_2(x) = 1 + 2x + 3(2x^2-1) = 1 + 2x + 6x^2 - 3 = -2 + 2x + 6x^2

    # Calculate its derivative
    deriv = poly.diff()  # 2 + 12x

    # Evaluate at a single point
    point = 0.5
    result = deriv(point)

    # Check that the result is correct
    # At x = 0.5: 2 + 12*0.5 = 2 + 6 = 8
    expected_result = 8.0
    assert np.isclose(result, expected_result), f"Result is not correct. Expected {expected_result}, got {result}"

    # Check that the result is a scalar, not a ChebyshevPolynomial
    assert np.isscalar(result), "Result is not a scalar"
