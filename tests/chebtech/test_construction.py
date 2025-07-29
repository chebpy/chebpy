"""Unit-tests for construction of Chebtech2 objects.

This module contains tests for the various ways to construct Chebtech2 objects,
including from values, coefficients, constants, and functions.
"""

import numpy as np
import pytest

from chebpy.core.chebtech import Chebtech2

from ..utilities import eps


# TODO: expand to all the constructor variants
def test_initvalues():
    """Test initialization of Chebtech2 objects from values.

    This test verifies that Chebtech2 objects can be correctly initialized
    from arrays of values, checking both empty arrays and arrays of various sizes.
    It ensures that the coefficients computed from the values match the expected
    coefficients computed directly from the values.
    """
    # test n = 0 case separately
    vals = np.random.rand(0)
    fun = Chebtech2.initvalues(vals)
    cfs = Chebtech2._vals2coeffs(vals)
    assert fun.coeffs.size == cfs.size == 0
    # now test the other cases
    for n in range(1, 10):
        vals = np.random.rand(n)
        fun = Chebtech2.initvalues(vals)
        cfs = Chebtech2._vals2coeffs(vals)
        assert np.max(fun.coeffs - cfs) == 0.0


def test_initidentity():
    """Test initialization of the identity function.

    This test verifies that the identity function created with initidentity
    correctly returns its input values when evaluated at random points.
    """
    x = Chebtech2.initidentity()
    s = -1 + 2 * np.random.rand(10000)
    assert np.max(s - x(s)) == 0.0


def test_coeff_construction():
    """Test construction of Chebtech2 objects from coefficients.

    This test verifies that Chebtech2 objects can be correctly initialized
    from arrays of coefficients, checking that the resulting object has the
    expected coefficients within machine precision.
    """
    coeffs = np.random.rand(10)
    f = Chebtech2(coeffs)
    assert isinstance(f, Chebtech2)
    assert np.max(f.coeffs - coeffs) < eps


def test_const_construction():
    """Test construction of constant Chebtech2 objects.

    This test verifies that constant Chebtech2 objects have the expected
    properties: size 1, isconst=True, isempty=False. It also checks that
    attempting to create a constant from a list raises a ValueError.
    """
    ff = Chebtech2.initconst(1.0)
    assert ff.size == 1
    assert ff.isconst
    assert not ff.isempty
    with pytest.raises(ValueError):
        Chebtech2.initconst([1.0])


def test_empty_construction():
    """Test construction of empty Chebtech2 objects.

    This test verifies that empty Chebtech2 objects have the expected
    properties: size 0, isconst=False, isempty=True. It also checks that
    attempting to create an empty Chebtech2 with arguments raises a TypeError.
    """
    ff = Chebtech2.initempty()
    assert ff.size == 0
    assert not ff.isconst
    assert ff.isempty
    with pytest.raises(TypeError):
        Chebtech2.initempty([1.0])


# Ensure reproducibility
np.random.seed(0)


# Test adaptive function construction
def test_adaptive_construction(testfunctions):
    """Test adaptive construction of Chebtech2 objects.

    This test verifies that Chebtech2 objects constructed adaptively
    from various test functions have sizes that are within expected
    bounds relative to the reference function length.

    Args:
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length, and _ is an unused parameter.
    """
    for fun, funlen, _ in testfunctions:
        ff = Chebtech2.initfun_adaptive(fun)
        assert ff.size - funlen <= 2
        assert ff.size - funlen > -1


# Test fixed length function construction
@pytest.mark.parametrize("n", [50, 500])
def test_fixedlen_construction(testfunctions, n):
    """Test fixed-length construction of Chebtech2 objects.

    This test verifies that Chebtech2 objects constructed with a fixed length
    from various test functions have exactly the specified size.

    Args:
        testfunctions: List of test functions, each represented as a tuple containing
            (fun, funlen, _) where fun is the function to test, funlen is the expected
            function length (unused in this test), and _ is an unused parameter.
        n: Fixed length to use for construction
    """
    for fun, _, _ in testfunctions:
        ff = Chebtech2.initfun_fixedlen(fun, n)
        assert ff.size == n
