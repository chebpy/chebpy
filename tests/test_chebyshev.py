import numpy as np
import pytest
from chebpy.core.chebyshev import ChebyshevPolynomial, from_coefficients, from_values, from_roots
from chebpy.core.utilities import Interval



@pytest.fixture
def valid_coeffs():
    """Fixture for valid coefficients."""
    return [1, 2, 3]


@pytest.fixture
def valid_interval():
    """Fixture for a valid interval."""
    return (-1.0, 1.0)


def test_post_init_converts_coeffs_to_numpy_array(valid_coeffs, valid_interval):
    """Test that __post_init__ converts coeffs to a numpy array."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert isinstance(poly.coef, np.ndarray), "Coeffs were not converted to numpy array."

def test_roots_with_valid_coefficients(valid_coeffs):
    """Test that roots method correctly computes the roots of a Chebyshev polynomial."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=(-1.0, 1.0))
    roots = poly.roots()
    assert isinstance(roots, np.ndarray), "Roots are not returned as a numpy array."
    assert len(roots) <= poly.degree(), "Number of roots exceeds the polynomial's degree."


def test_degree_correctly_calculates(valid_coeffs, valid_interval):
    """Test that ChebyshevPolynomial degree property returns the correct degree."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert poly.degree() == len(valid_coeffs) - 1, "Degree of polynomial is computed incorrectly."


def test_eq_returns_true_for_equal_polynomials(valid_coeffs, valid_interval):
    """Test that __eq__ returns True for equal polynomials."""
    poly1 = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    poly2 = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert poly1 == poly2, "__eq__ failed to identify equal polynomials."


def test_eq_returns_false_for_different_coeffs(valid_interval):
    """Test that __eq__ returns False for polynomials with different coefficients."""
    poly1 = ChebyshevPolynomial(coef=[1, 2, 3], domain=valid_interval)
    poly2 = ChebyshevPolynomial(coef=[4, 5, 6], domain=valid_interval)
    assert poly1 != poly2, "__eq__ failed to identify polynomials with different coefficients as unequal."


def test_eq_returns_false_for_different_domains(valid_coeffs):
    """Test that __eq__ returns False for polynomials defined on different domains."""
    poly1 = ChebyshevPolynomial(coef=valid_coeffs, domain=(-1.0, 1.0))
    poly2 = ChebyshevPolynomial(coef=valid_coeffs, domain=(0.0, 1.0))
    assert poly1 != poly2, "__eq__ failed to identify polynomials with different domains as unequal."


def test_eq_returns_false_for_non_chebyshev_objects(valid_coeffs, valid_interval):
    """Test that __eq__ returns False when compared to a non-Chebyshev object."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert poly != "not a polynomial", "__eq__ failed to return False for a non-Chebyshev object."


def test_call_evaluates_polynomial_at_single_point(valid_coeffs, valid_interval):
    """Test that __call__ evaluates the polynomial at a single point."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    result = poly(0.5)
    assert isinstance(result, float), "Evaluated result is not a float."


def test_call_evaluates_polynomial_at_multiple_points(valid_coeffs, valid_interval):
    """Test that __call__ evaluates the polynomial at multiple points."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    points = np.linspace(-1.0, 1.0, 10)
    results = poly(points)
    assert isinstance(results, np.ndarray), "Evaluated results are not a numpy array."
    assert results.shape == points.shape, "Shape of evaluated results does not match input points shape."

    def test_roots_returns_correct_roots(valid_coeffs):

        """Test that roots method correctly computes the roots of the Chebyshev polynomial."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=(-1.0, 1.0))
    roots = poly.roots()
    assert isinstance(roots, np.ndarray), "Roots are not returned as a numpy array."
    assert len(roots) <= poly.degree(), "Number of roots exceeds the polynomial's degree."


def test_call_handles_out_of_domain_points(valid_coeffs, valid_interval):
    """Test that __call__ handles evaluation of points outside the domain."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    result = poly(2.0)  # Point outside the interval [-1, 1]
    assert isinstance(result, float), "Evaluated result for out-of-domain point is not a float."


def test_call_handles_empty_input(valid_coeffs):
    """Test that __call__ handles an empty array as input."""
    poly = ChebyshevPolynomial(coef=valid_coeffs)
    points = np.array([])
    results = poly(points)
    assert isinstance(results, np.ndarray), "Results for empty input are not a numpy array."
    assert results.size == 0, "Results for empty input are not empty."


def test_post_init_converts_interval_to_Interval(valid_coeffs, valid_interval):
    """Test that __post_init__ converts interval to an Interval object."""
    poly = ChebyshevPolynomial(coef=valid_coeffs, domain=valid_interval)
    assert isinstance(poly.domain, np.ndarray), "Interval was not converted to numpy array."


def test_post_init_raises_error_on_invalid_interval(valid_coeffs):
    """Test that __post_init__ raises an error when interval is invalid."""
    invalid_interval = (1.0, -1.0)
    with pytest.raises(Exception):
        ChebyshevPolynomial(coeffs=valid_coeffs, interval=invalid_interval)


def test_post_init_handles_numpy_array_coeffs():
    """Test that __post_init__ works when coeffs is already a numpy array."""
    coeffs_np = np.array([4, 5, 6])
    poly = ChebyshevPolynomial(coef=coeffs_np)
    assert isinstance(poly.coef, np.ndarray), "Coeffs as numpy array were altered unexpectedly."
    assert np.array_equal(poly.coef, coeffs_np), "Coeffs were modified unexpectedly."


# Tests for factory functions
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
