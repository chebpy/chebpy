"""Tests for the roots, companion matrices, and eigenvalues of ChebyshevPolynomial."""

import numpy as np
import numpy.linalg as la
import numpy.polynomial.chebyshev as cheb

from chebpy.core.chebyshev import ChebyshevPolynomial, from_roots


def test_roots_of_quadratic_polynomial():
    """Test that roots method correctly computes the roots of a quadratic polynomial."""
    # Create a quadratic polynomial with known roots at -1 and 1
    # In standard form: x^2 - 1 = 0
    # In Chebyshev basis: T_0(x) - T_2(x) = 0
    poly = ChebyshevPolynomial(coef=[1, 0, -1])

    # Compute the roots
    roots = poly.roots()

    # Check that the roots are correct
    assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}"
    assert np.allclose(sorted(roots), [-1, 1]), f"Expected roots at -1 and 1, got {sorted(roots)}"


def test_roots_of_cubic_polynomial():
    """Test that roots method correctly computes the roots of a cubic polynomial."""
    # Create a cubic polynomial with known roots at -1, 0, and 1
    # In standard form: x^3 - x = 0 or x(x^2 - 1) = 0
    # In Chebyshev basis: this is more complex, so we'll create it from roots
    poly = from_roots([-1, 0, 1])

    # Compute the roots
    roots = poly.roots()

    # Check that the roots are correct
    assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}"
    assert np.allclose(sorted(roots), [-1, 0, 1]), f"Expected roots at -1, 0, and 1, got {sorted(roots)}"


def test_roots_of_polynomial_with_multiple_roots():
    """Test that roots method correctly computes the roots of a polynomial with multiple roots."""
    # Create a polynomial with a double root at 0 and a single root at 1
    # In standard form: x^2 * (x - 1) = 0
    # In Chebyshev basis: this is more complex, so we'll create it from roots
    poly = from_roots([0, 0, 1])

    # Compute the roots
    roots = poly.roots()

    # Check that the roots are correct
    assert len(roots) == 3, f"Expected 3 roots, got {len(roots)}"
    # The roots might not be exactly [0, 0, 1] due to numerical issues,
    # but they should be close
    sorted_roots = sorted(roots)
    assert np.isclose(sorted_roots[0], 0, atol=1e-7), f"Expected root at 0, got {sorted_roots[0]}"
    assert np.isclose(sorted_roots[1], 0, atol=1e-7), f"Expected root at 0, got {sorted_roots[1]}"
    assert np.isclose(sorted_roots[2], 1, atol=1e-7), f"Expected root at 1, got {sorted_roots[2]}"


def test_roots_of_polynomial_with_complex_roots():
    """Test that roots method correctly computes the complex roots of a polynomial."""
    # Create a polynomial with complex roots at 1+1j and 1-1j
    # In standard form: (x - (1+1j)) * (x - (1-1j)) = 0
    # This expands to x^2 - 2x + 2 = 0
    # In Chebyshev basis: this is more complex, so we'll create it from roots
    poly = from_roots([1 + 1j, 1 - 1j])

    # Compute the roots
    roots = poly.roots()

    # Check that the roots are correct
    assert len(roots) == 2, f"Expected 2 roots, got {len(roots)}"

    # The roots might be in any order, so we need to check that both expected roots are present
    expected_roots = [1 + 1j, 1 - 1j]
    for root in roots:
        # Check if the root is close to any of the expected roots
        assert any(np.isclose(root, expected_root, atol=1e-7) for expected_root in expected_roots), (
            f"Root {root} is not close to any of the expected roots {expected_roots}"
        )


def test_companion_matrix_of_quadratic_polynomial():
    """Test the computation of the companion matrix for a quadratic polynomial."""
    # Create a quadratic polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3])

    # Compute the companion matrix using numpy's chebcompanion function
    companion = cheb.chebcompanion(poly.coef)

    # Check that the companion matrix has the correct shape
    assert companion.shape == (poly.degree(), poly.degree()), (
        f"Expected shape {(poly.degree(), poly.degree())}, got {companion.shape}"
    )

    # Check that the eigenvalues of the companion matrix are the roots of the polynomial
    eigenvalues = la.eigvals(companion)
    roots = poly.roots()

    # Sort both arrays for comparison
    eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
    roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))

    # Check that the eigenvalues match the roots
    assert np.allclose(eigenvalues, roots), f"Eigenvalues {eigenvalues} do not match roots {roots}"


def test_companion_matrix_of_cubic_polynomial():
    """Test the computation of the companion matrix for a cubic polynomial."""
    # Create a cubic polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3, 4])

    # Compute the companion matrix using numpy's chebcompanion function
    companion = cheb.chebcompanion(poly.coef)

    # Check that the companion matrix has the correct shape
    assert companion.shape == (poly.degree(), poly.degree()), (
        f"Expected shape {(poly.degree(), poly.degree())}, got {companion.shape}"
    )

    # Check that the eigenvalues of the companion matrix are the roots of the polynomial
    eigenvalues = la.eigvals(companion)
    roots = poly.roots()

    # Sort both arrays for comparison
    eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
    roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))

    # Check that the eigenvalues match the roots
    assert np.allclose(eigenvalues, roots), f"Eigenvalues {eigenvalues} do not match roots {roots}"


def test_eigenvalues_of_companion_matrix():
    """Test that the eigenvalues of the companion matrix are the roots of the polynomial."""
    # Create a polynomial with known roots
    roots_expected = [-1, 0, 1]
    poly = from_roots(roots_expected)

    # Compute the companion matrix
    companion = cheb.chebcompanion(poly.coef)

    # Compute the eigenvalues of the companion matrix
    eigenvalues = la.eigvals(companion)

    # Sort both arrays for comparison
    eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))
    roots_expected = sorted(roots_expected)

    # Check that the eigenvalues match the expected roots
    assert np.allclose(eigenvalues, roots_expected), (
        f"Eigenvalues {eigenvalues} do not match expected roots {roots_expected}"
    )


def test_eigenvalues_of_companion_matrix_with_complex_roots():
    """Test that the eigenvalues of the companion matrix are the complex roots of the polynomial."""
    # Create a polynomial with known complex roots
    roots_expected = [1 + 1j, 1 - 1j]
    poly = from_roots(roots_expected)

    # Compute the companion matrix
    companion = cheb.chebcompanion(poly.coef)

    # Compute the eigenvalues of the companion matrix
    eigenvalues = la.eigvals(companion)

    # The eigenvalues might be in any order, so we need to check that both expected roots are present
    for eigenvalue in eigenvalues:
        # Check if the eigenvalue is close to any of the expected roots
        assert any(np.isclose(eigenvalue, root_expected, atol=1e-7) for root_expected in roots_expected), (
            f"Eigenvalue {eigenvalue} is not close to any of the expected roots {roots_expected}"
        )


def test_roots_companion_eigenvalues_relationship():
    """Test the relationship between roots, companion matrix, and eigenvalues."""
    # Create a polynomial
    poly = ChebyshevPolynomial(coef=[1, 2, 3, 4, 5])

    # Compute the roots directly
    roots = poly.roots()

    # Compute the companion matrix
    companion = cheb.chebcompanion(poly.coef)

    # Compute the eigenvalues of the companion matrix
    eigenvalues = la.eigvals(companion)

    # Sort both arrays for comparison
    roots = sorted(roots, key=lambda x: (np.real(x), np.imag(x)))
    eigenvalues = sorted(eigenvalues, key=lambda x: (np.real(x), np.imag(x)))

    # Check that the roots match the eigenvalues
    assert np.allclose(roots, eigenvalues), f"Roots {roots} do not match eigenvalues {eigenvalues}"
