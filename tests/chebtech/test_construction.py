"""Unit-tests for construction of Chebtech objects.

This module contains tests for the various ways to construct Chebtech objects,
including from values, coefficients, constants, and functions.
"""

import numpy as np
import pytest

from chebpy.chebtech import Chebtech

from ..utilities import eps

# Ensure reproducibility
rng = np.random.default_rng(0)


# TODO: expand to all the constructor variants
def test_initvalues():
    """Test initialization of Chebtech objects from values.

    This test verifies that Chebtech objects can be correctly initialized
    from arrays of values, checking both empty arrays and arrays of various sizes.
    It ensures that the coefficients computed from the values match the expected
    coefficients computed directly from the values.
    """
    # test n = 0 case separately
    vals = rng.random(0)
    fun = Chebtech.initvalues(vals)
    cfs = Chebtech._vals2coeffs(vals)
    assert fun.coeffs.size == cfs.size == 0
    # now test the other cases
    for n in range(1, 10):
        vals = rng.random(n)
        fun = Chebtech.initvalues(vals)
        cfs = Chebtech._vals2coeffs(vals)
        assert np.max(np.abs(fun.coeffs - cfs)) == 0.0


def test_initidentity():
    """Test initialization of the identity function.

    This test verifies that the identity function created with initidentity
    correctly returns its input values when evaluated at random points.
    """
    x = Chebtech.initidentity()
    s = -1 + 2 * rng.random(10000)
    assert np.max(np.abs(s - x(s))) == 0.0


def test_coeff_construction():
    """Test construction of Chebtech objects from coefficients.

    This test verifies that Chebtech objects can be correctly initialized
    from arrays of coefficients, checking that the resulting object has the
    expected coefficients within machine precision.
    """
    coeffs = rng.random(10)
    f = Chebtech(coeffs)
    assert isinstance(f, Chebtech)
    assert np.max(np.abs(f.coeffs - coeffs)) < eps


# Test adaptive function construction
def test_adaptive_construction(testfunctions):
    """Test adaptive construction of Chebtech objects.

    This test verifies that Chebtech objects constructed adaptively
    from various test functions have sizes that are within expected
    bounds relative to the reference function length.

    Args:
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length, and _ is an unused parameter.
    """
    for fun, funlen, _ in testfunctions:
        ff = Chebtech.initfun_adaptive(fun)
        assert ff.size - funlen <= 2
        assert ff.size - funlen > -1


# Test fixed length function construction
@pytest.mark.parametrize("n", [50, 500])
def test_fixedlen_construction(testfunctions, n):
    """Test fixed-length construction of Chebtech objects.

    This test verifies that Chebtech objects constructed with a fixed length
    from various test functions have exactly the specified size.

    Args:
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length (unused in this test), and _ is an unused parameter.
        n: Fixed length to use for construction
    """
    for fun, _, _ in testfunctions:
        ff = Chebtech.initfun_fixedlen(fun, n)
        assert ff.size == n


def test_initconst_with_non_scalar():
    """Test that initconst raises ValueError for non-scalar input."""
    with pytest.raises(ValueError):
        Chebtech.initconst([1, 2, 3])

    with pytest.raises(ValueError):
        Chebtech.initconst(np.array([1, 2, 3]))


def test_imag_complex_chebtech():
    """Test the imag() method on a complex Chebtech."""
    # Create a complex Chebtech
    f = Chebtech.initfun_adaptive(lambda x: np.exp(1j * np.pi * x))

    # Verify it's complex
    assert f.iscomplex

    # Get the imaginary part
    imag_f = f.imag()

    # Check that it's a Chebtech
    assert isinstance(imag_f, Chebtech)

    # Check that it gives the correct values at sample points
    xx = np.linspace(-1, 1, 100)
    expected = np.imag(np.exp(1j * np.pi * xx))
    actual = imag_f(xx)
    assert np.max(np.abs(actual - expected)) < 1e-10


def test_chebtech_with_nan_coefficients():
    """Test that evaluating a Chebtech with NaN coefficients returns NaN."""
    # Create a Chebtech with NaN coefficients
    nan_coeffs = np.array([1.0, np.nan, 2.0])
    f = Chebtech(nan_coeffs)

    # Evaluate at some points
    xx = np.array([-1.0, 0.0, 1.0])
    result = f(xx)

    # Check that all results are NaN
    assert np.all(np.isnan(result)), "Expected all NaN values when coefficients contain NaN"
