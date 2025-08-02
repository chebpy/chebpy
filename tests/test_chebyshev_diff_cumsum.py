"""Tests for the comprehensive functionality of Chebyshev polynomials."""

import numpy as np

from chebpy.core.chebyshev import ChebyshevPolynomial, from_constant


def test_diff_calculates_derivative_correctly():
    """Test that diff correctly calculates the derivative of a polynomial."""
    # Create a polynomial with known coefficients
    poly = ChebyshevPolynomial(coef=[1, 2, 3])  # 1 + 2*T_1(x) + 3*T_2(x)

    # Calculate its derivative
    deriv = poly.diff()

    # Check at specific points
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        # Calculate the derivative analytically
        # For a Chebyshev polynomial 1 + 2*T_1(x) + 3*T_2(x),
        # the derivative is 2 + 12*x
        expected = 2 + 12 * x

        # Get the value from the derivative polynomial
        actual = deriv(x)

        # Check that they match
        assert abs(actual - expected) < 1e-10, f"Derivative at {x} is {actual}, expected {expected}"


def test_diff_of_constant_polynomial():
    """Test that the derivative of a constant polynomial is zero."""
    # Create a constant polynomial
    poly = from_constant(42.0)

    # Calculate its derivative
    deriv = poly.diff()

    # Check that it's zero at various points
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(deriv(x)) < 1e-10, f"Derivative of constant at {x} is not zero: {deriv(x)}"


def test_cumsum_calculates_antiderivative_correctly():
    """Test that cumsum correctly calculates the antiderivative of a polynomial."""
    # Create a polynomial with known coefficients
    poly = ChebyshevPolynomial(coef=[0, 1])  # T_1(x) = x

    # Calculate its antiderivative
    integ = poly.cumsum()

    # Check at specific points
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        # Calculate the antiderivative analytically
        # For T_1(x) = x, the antiderivative with lower bound -1 is (x^2 - 1)/2
        expected = (x**2 - 1) / 2

        # Get the value from the antiderivative polynomial
        actual = integ(x)

        # Check that they match
        assert abs(actual - expected) < 1e-10, f"Antiderivative at {x} is {actual}, expected {expected}"


def test_cumsum_of_zero_polynomial():
    """Test that the antiderivative of a zero polynomial is zero."""
    # Create a zero polynomial
    poly = ChebyshevPolynomial(coef=[0])

    # Calculate its antiderivative
    integ = poly.cumsum()

    # Check that it's zero at various points
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        assert abs(integ(x)) < 1e-10, f"Antiderivative of zero at {x} is not zero: {integ(x)}"


def test_round_trip_diff_cumsum():
    """Test that taking the derivative of an antiderivative gives back the original function."""
    # Create a polynomial with known coefficients
    poly = ChebyshevPolynomial(coef=[1, 2, 3])

    # Take the antiderivative and then the derivative
    round_trip = poly.cumsum().diff()

    # Check at specific points
    x_values = np.linspace(-1, 1, 10)
    for x in x_values:
        # Get the values from both polynomials
        original = poly(x)
        result = round_trip(x)

        # Check that they match
        assert abs(result - original) < 1e-10, f"Round trip at {x} is {result}, expected {original}"
