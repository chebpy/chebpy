"""Tests for complex ChebyshevPolynomial functionality."""

import numpy as np

from chebpy.core.chebyshev import ChebyshevPolynomial


def test_iscomplex_property_with_real_coefficients():
    """Test that iscomplex property returns False for polynomials with real coefficients."""
    poly = ChebyshevPolynomial(coef=[1, 2, 3])
    assert not poly.iscomplex, "iscomplex property should be False for real coefficients."


def test_iscomplex_property_with_complex_coefficients():
    """Test that iscomplex property returns True for polynomials with complex coefficients."""
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
    assert poly.iscomplex, "iscomplex property should be True for complex coefficients."


def test_real_method_with_real_polynomial():
    """Test that real method returns the original polynomial for real polynomials."""
    poly = ChebyshevPolynomial(coef=[1, 2, 3])
    real_poly = poly.real()

    # Check that it's the same instance (not a copy)
    assert real_poly is poly, "real method should return the original polynomial for real polynomials."

    # Check that the coefficients are the same
    assert np.array_equal(real_poly.coef, poly.coef), "Coefficients should be unchanged."


def test_real_method_with_complex_polynomial():
    """Test that real method returns a new polynomial with real coefficients for complex polynomials."""
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
    real_poly = poly.real()

    # Check that it's a new instance
    assert real_poly is not poly, "real method should return a new polynomial for complex polynomials."

    # Check that the coefficients are the real part of the original coefficients
    assert np.array_equal(real_poly.coef, np.real(poly.coef)), "Coefficients should be the real part of the original."

    # Check that the domain and window are preserved
    assert np.array_equal(real_poly.domain, poly.domain), "Domain should be preserved."
    assert np.array_equal(real_poly.window, poly.window), "Window should be preserved."

    # Check that the symbol is preserved
    assert real_poly.symbol == poly.symbol, "Symbol should be preserved."


def test_imag_method_with_real_polynomial():
    """Test that imag method returns the original polynomial for real polynomials."""
    poly = ChebyshevPolynomial(coef=[1, 2, 3])
    imag_poly = poly.imag()

    # Check that it's the same instance (not a copy)
    assert imag_poly is poly, "imag method should return the original polynomial for real polynomials."

    # Check that the coefficients are the same
    assert np.array_equal(imag_poly.coef, poly.coef), "Coefficients should be unchanged."


def test_imag_method_with_complex_polynomial():
    """Test that imag method returns a new polynomial with imaginary coefficients for complex polynomials."""
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])
    imag_poly = poly.imag()

    # Check that it's a new instance
    assert imag_poly is not poly, "imag method should return a new polynomial for complex polynomials."

    # Check that the coefficients are the imaginary part of the original coefficients
    assert np.array_equal(imag_poly.coef, np.imag(poly.coef)), "Coefficients should be the imaginary part."

    # Check that the domain and window are preserved
    assert np.array_equal(imag_poly.domain, poly.domain), "Domain should be preserved."
    assert np.array_equal(imag_poly.window, poly.window), "Window should be preserved."

    # Check that the symbol is preserved
    assert imag_poly.symbol == poly.symbol, "Symbol should be preserved."


def test_evaluate_complex_polynomial():
    """Test evaluating a complex polynomial at a point."""
    # Create a complex polynomial
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])

    # Evaluate at a point
    x = 0.5
    result = poly(x)

    # Calculate the expected result manually
    # For a Chebyshev polynomial with coefficients [a, b, c], the value at x is:
    # a * T_0(x) + b * T_1(x) + c * T_2(x) = a + b * x + c * (2x^2 - 1)
    expected_real = 1 + 2 * x + 3 * (2 * x**2 - 1)
    expected_imag = 1 + 2 * x + 3 * (2 * x**2 - 1)
    expected = complex(expected_real, expected_imag)

    # Check that the result is correct
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def test_real_plus_imag_equals_original():
    """Test that real + i*imag equals the original complex polynomial."""
    # Create a complex polynomial
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])

    # Get the real and imaginary parts
    real_poly = poly.real()
    imag_poly = poly.imag()

    # Create a new complex polynomial from the real and imaginary parts
    reconstructed = real_poly + 1j * imag_poly

    # Check that the reconstructed polynomial has the same coefficients as the original
    assert np.allclose(reconstructed.coef, poly.coef), "Reconstructed polynomial should match the original."

    # Check that the domain and window are preserved
    assert np.array_equal(reconstructed.domain, poly.domain), "Domain should be preserved."
    assert np.array_equal(reconstructed.window, poly.window), "Window should be preserved."


def test_real_and_imag_evaluate_correctly():
    """Test that real and imag parts evaluate correctly."""
    # Create a complex polynomial
    poly = ChebyshevPolynomial(coef=[1 + 1j, 2 + 2j, 3 + 3j])

    # Get the real and imaginary parts
    real_poly = poly.real()
    imag_poly = poly.imag()

    # Evaluate at several points
    points = np.linspace(-1, 1, 10)
    for x in points:
        # Evaluate the original polynomial
        result = poly(x)

        # Evaluate the real and imaginary parts
        real_result = real_poly(x)
        imag_result = imag_poly(x)

        # Check that the real and imaginary parts match
        assert np.isclose(real_result, np.real(result)), f"Real part mismatch at {x}"
        assert np.isclose(imag_result, np.imag(result)), f"Imaginary part mismatch at {x}"
